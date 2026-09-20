"""Per-node EN / 書面粵語 translation and Cantonese register checks."""

from __future__ import annotations

import re
from pathlib import Path

from book_combiner.llm.base import LLMClient
from book_combiner.merge import (
    DEFAULT_CHARS_PER_TOKEN,
    MODEL_OUTPUT_CAP,
    atomic_write_text,
    load_token_stats,
    max_tokens_for,
    needed_tokens,
)
from book_combiner.outline import fill_template, sha256_file, sha256_text, split_prompt_sections

TRANSLATE_TEMPERATURE = 0.2
EN_CHAR_FACTOR = 1.6
YUE_CHAR_FACTOR = 1.1
HAN_DENSITY_CAP = 0.005
YUE_PARTICLE_RATIO_CAP = 3.0
YUE_PARTICLES_PER_HAN = 500
STAGE_EN = "translate-en"
STAGE_YUE = "translate-yue"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
TRANSLATE_EN_PROMPT_PATH = _PROMPTS_DIR / "translate_en.txt"
TRANSLATE_YUE_PROMPT_PATH = _PROMPTS_DIR / "translate_yue.txt"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
YAML_FRONT_RE = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n?", re.S)
SOURCE_TAG_RE = re.compile(r"\[[^\]\n]+#[^\]\n]+\]")
PAREN_RE = re.compile(r"\([^)]*\)|（[^）]*）")
LATIN_RE = re.compile(r"[A-Za-z]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
H4_SPLIT_RE = re.compile(r"(?=^#### )", re.M)
BOOK_TITLE_RE = re.compile(r"《[^》]{1,80}》")

# Phrase replacements applied before the character map (longest first).
TECH_TERM_S2T: tuple[tuple[str, str], ...] = (
    ("微反应", "微反應"),
    ("额肌", "額肌"),
    ("皱眉肌", "皺眉肌"),
    ("眼轮匝肌", "眼輪匝肌"),
    ("口轮匝肌", "口輪匝肌"),
)

# Simplified → Traditional pairs. Only characters that actually differ are listed.
_S2T_PAIRS = (
    "们們这這过過来來对對时時会會么麼为為无無发發国國说說经經现現开開关關还還"
    "进進从從长長与與问問间間个個学學后後点點电電车車书書语語词詞觉覺变變"
    "灵靈称稱应應仅僅并並见見让讓该該认認识識调調议議论論记記证證护護报報"
    "体體机機动動当當样樣种種头頭条條儿兒两兩产產内內务務号號处處战戰东東"
    "声聲气氣页頁门門贝貝马馬风風云雲广廣厂廠飞飛习習乐樂乔喬买買乱亂"
    "亚亞亲親亿億仑侖仓倉仪儀价價众眾优優伙夥伞傘伟偉传傳伤傷伦倫伪偽侦偵"
    "侧側侨僑余餘侠俠侣侶俭儉债債倾傾刚剛创創刘劉则則别別刹剎剂劑剑劍"
    "剧劇劝勸办辦劲勁劳勞势勢匀勻区區医醫华華协協单單卖賣卫衛卷捲厅廳"
    "历歷厉厲压壓厌厭厕廁厘釐县縣参參双雙叠疊叹嘆叽嘰吁籲吨噸听聽启啟吴吳"
    "呕嘔呗唄员員呛嗆呜嗚周週咸鹹响響哑啞哗嘩哟喲唤喚喷噴喽嘍嘱囑严嚴团團"
    "园園围圍图圖圆圓圣聖场場坏壞块塊坚堅坛壇垒壘堕墮壮壯夸誇夹夾夺奪奋奮"
    "奖獎妆妝妇婦妈媽娄婁娱娛检檢樱櫻欢歡残殘殴毆杀殺杂雜权權杆桿杨楊杰傑"
    "松鬆极極构構枢樞枪槍枫楓柜櫃树樹栖棲档檔桥橋梦夢椭橢横橫欲慾岁歲归歸"
    "壳殼汉漢汤湯汹洶沟溝没沒沥瀝沦淪泪淚泻瀉泽澤洁潔洒灑浅淺浆漿浇澆浊濁"
    "测測济濟浑渾浓濃涂塗涛濤润潤涧澗涨漲渊淵渍漬渐漸渔漁渗滲湾灣湿濕满滿"
    "滥濫滨濱滩灘澜瀾濑瀨濒瀕灭滅灯燈灾災炉爐炼煉烁爍烂爛烦煩烧燒热熱焕煥"
    "烫燙爱愛爷爺牵牽牺犧状狀犹猶独獨狮獅狱獄猎獵猪豬猫貓献獻玛瑪环環琐瑣"
    "琼瓊瑶瑤画畫畅暢畴疇疗療疯瘋疮瘡痒癢痴癡瘫癱癣癬皱皺监監盖蓋盘盤眯瞇"
    "着著睁睜睐睞瞒瞞瞩矚矫矯码碼砖磚砚硯础礎硕碩确確碍礙碱鹼礼禮祷禱祸禍"
    "禅禪离離积積税稅稳穩穷窮窃竊窝窩窑窯竞競笔筆筑築筛篩筝箏筹籌签簽简簡"
    "篮籃类類粮糧紧緊纠糾红紅纤纖约約级級纪紀纯純纲綱纳納纵縱纷紛纸紙"
    "纹紋纺紡纽紐线線练練组組细細织織终終绊絆绍紹绑綁绒絨结結绕繞绘繪给給"
    "络絡绝絕绞絞统統绢絹绣繡继繼绩績绪緒续續绰綽绳繩维維绵綿绷繃绸綢综綜"
    "绽綻绿綠缀綴缅緬缆纜缉緝缎緞缓緩缔締缕縷编編缘緣缚縛缝縫缠纏缨纓缩縮"
    "缭繚缰韁网網罗羅罚罰罢罷闻聞联聯聪聰肃肅肠腸肤膚肾腎肿腫胀脹胁脅胆膽"
    "胜勝胡鬍胧朧胫脛胶膠脉脈脏髒脑腦脚腳脱脫脸臉腊臘腾騰舍捨舰艦舱艙艰艱"
    "艳艷艺藝节節芜蕪苍蒼苏蘇苹蘋范範茎莖茧繭荆荊荐薦荚莢荡蕩荣榮荤葷荧熒"
    "药藥莅蒞莱萊莲蓮获獲莹瑩莺鶯莼蓴萝蘿萤螢营營萧蕭萨薩葱蔥蒋蔣蓝藍蓦驀"
    "蔼藹蕴蘊蓟薊藓蘚虑慮虚虛虫蟲虽雖虾蝦蚀蝕蚁蟻蚂螞蚕蠶蚬蜆蚝蠔蛮蠻蛱蛺"
    "蜕蛻蜗蝸蜡蠟蝇蠅蝉蟬蝎蠍螨蟎衅釁衔銜补補衬襯袜襪袭襲装裝裆襠裤褲褛褸"
    "观觀规規觅覓视視览覽觊覬觌覿觎覦觐覲觑覷觞觴触觸订訂计計讯訊讨討训訓"
    "讲講讳諱讴謳讶訝许許讹訛讼訟讽諷设設访訪诀訣评評诅詛诈詐诉訴诊診诏詔"
    "译譯试試诗詩诘詰诚誠诛誅话話诞誕诡詭询詢诣詣详詳诫誡诬誣误誤诱誘诲誨"
    "诵誦请請诸諸诺諾读讀诽誹课課诿諉谁誰谄諂谅諒谆諄谈談谊誼谋謀谍諜谏諫"
    "谐諧谑謔谒謁谓謂谕諭谗讒谘諮谙諳谚諺谛諦谜謎谟謨谢謝谣謠谤謗谥謚谦謙"
    "谧謐谨謹谩謾谪謫谬謬谭譚谮譖谯譙谱譜谲譎谳讞谴譴谵譫谶讖贝貝贞貞负負"
    "贡貢财財责責贤賢败敗账賬货貨质質贩販贪貪贫貧贬貶购購贮貯贯貫贰貳贱賤"
    "贲賁贴貼贵貴贷貸贸貿费費贺賀贻貽贼賊贾賈贿賄赀貲赁賃赂賂赃贓资資赅賅"
    "赋賦赌賭赍齎赎贖赏賞赐賜赓賡赔賠赖賴赘贅赚賺赛賽赜賾赞贊赠贈赡贍赢贏"
    "赶趕趋趨跃躍践踐跷蹺跻躋踊踴踪蹤踯躑蹑躡躏躪躯軀轧軋轨軌轩軒轫軔转轉"
    "轮輪软軟轰轟轴軸轻輕载載轿轎较較辅輔辆輛辈輩辉輝辍輟辐輻辑輯输輸辕轅"
    "辖轄辗輾辙轍辞辭辟闢辩辯辫辮边邊辽遼达達迁遷迈邁运運远遠违違连連迟遲"
    "迹跡适適选選逊遜递遞游遊遗遺邓鄧邮郵邻鄰郁鬱郑鄭郸鄲酝醞酿釀采採释釋"
    "里裡鉴鑒针針钉釘钓釣钗釵钙鈣钝鈍钞鈔钟鐘钠鈉钢鋼钥鑰钦欽钨鎢钩鉤钮鈕"
    "钱錢钳鉗钵缽钻鑽钼鉬钾鉀铀鈾铁鐵铂鉑铃鈴铅鉛铆鉚铜銅铝鋁铠鎧铡鍘铣銑"
    "铭銘铲鏟银銀铸鑄铺鋪链鏈销銷锁鎖锄鋤锅鍋锈鏽锋鋒锌鋅锐銳锑銻锗鍺错錯"
    "锚錨锡錫锣鑼锤錘锥錐锦錦锨杴锭錠键鍵锯鋸锰錳锹鍬锻鍛镀鍍镁鎂镇鎮镊鑷"
    "镍鎳镐鎬镜鏡镣鐐镰鐮镶鑲闪閃闭閉闯闖闰閏闲閒闷悶闸閘闹鬧闺閨闽閩阀閥"
    "阁閣阅閱阉閹阎閻阐闡阑闌阔闊阙闕队隊阳陽阴陰阵陣阶階际際陆陸陈陳陕陝"
    "陨隕险險随隨隐隱隶隸难難雇僱雏雛雾霧雳靂霉黴静靜韩韓韦韋韧韌韬韜韵韻"
    "顶頂顷頃项項顺順须須顽頑顾顧顿頓颁頒颂頌预預颅顱领領颇頗颈頸颉頡颊頰"
    "颌頜颍潁频頻颓頹颖穎颗顆题題颚顎颛顓颜顏额額颠顛颢顥颤顫颦顰飒颯飓颶"
    "飘飄飙飆饥飢饭飯饮飲饯餞饰飾饱飽饲飼饴飴饵餌饶饒饺餃饼餅饿餓馁餒馅餡"
    "馆館馈饋馊餿馋饞驭馭驮馱驯馴驰馳驱驅驳駁驴驢驶駛驷駟驹駒驻駐驼駝驾駕"
    "驿驛骂罵骄驕骆駱骇駭骈駢骋騁验驗骏駿骑騎骗騙骚騷骛騖骜驁骞騫骠驃骡騾"
    "骤驟骥驥髅髏鬓鬢魇魘鱼魚鲁魯鲜鮮鲤鯉鲨鯊鲫鯽鲸鯨鳃鰓鳄鱷鳍鰭鳖鱉鳞鱗"
    "鸟鳥鸠鳩鸡雞鸣鳴鸿鴻鸽鴿鹃鵑鹅鵝鹊鵲鹏鵬鹤鶴鹰鷹卤鹵盐鹽麦麥麸麩黄黃"
    "党黨齐齊齿齒龄齡龙龍龟龜罗羅"
    "兴興属屬温溫显顯术術态態据據总總标標准準录錄构構扩擴扫掃护護"
    "摆擺扩擴拥擁择擇摄攝摇搖摊攤撑撐掷擲挥揮损損换換据據捷捷"
)


def _build_s2t() -> tuple[dict[str, str], frozenset[str]]:
    mapping: dict[str, str] = {}
    chars = "".join(_S2T_PAIRS)
    for i in range(0, len(chars) - 1, 2):
        src, dst = chars[i], chars[i + 1]
        if src != dst:
            mapping.setdefault(src, dst)
    return mapping, frozenset(mapping)


S2T_MAP, _S2T_KEYS = _build_s2t()
S2T_TRANS = str.maketrans(S2T_MAP)
# Codepoints that exist in Traditional running text. S2T_MAP may still rewrite
# them (里→裡, 后→後, 松→鬆) but they are not Simplified-only (公里/皇后/松樹).
_TRADITIONAL_SHARED = frozenset(
    "里后松云准余采干面只台向克卜占斗术刮御系冲胡划卷丰沈姜于并咸蒙"
)
# Simplified-only denylist: seed examples plus S2T keys that are not shared Traditional.
SIMPLIFIED_ONLY = (_S2T_KEYS | frozenset("们这过来对时会么为无发")) - _TRADITIONAL_SHARED


class TranslateError(Exception):
    """Translation stage failed (missing artifacts or unrecoverable node)."""


def strip_yaml_front_matter(text: str) -> str:
    return YAML_FRONT_RE.sub("", text, count=1)


def traditionalize(text: str) -> str:
    """Map Simplified CJK to Traditional; apply the translate-yue technical-term list first."""
    out = text
    for src, dst in TECH_TERM_S2T:
        out = out.replace(src, dst)
    return out.translate(S2T_TRANS)


def body_for_cantonese_eval(text: str) -> str:
    """Body only: drop YAML, source/conflict tags, and Latin."""
    body = strip_yaml_front_matter(text)
    body = SOURCE_TAG_RE.sub("", body)
    body = LATIN_RE.sub("", body)
    return body


def han_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def simplified_codepoints(text: str) -> list[str]:
    seen: list[str] = []
    found: set[str] = set()
    for ch in text:
        if ch in SIMPLIFIED_ONLY and ch not in found:
            found.add(ch)
            seen.append(ch)
    return seen


def cantonese_particle_counts(text: str) -> tuple[int, int]:
    mandarin = text.count("的") + text.count("是") + text.count("不")
    yue = text.count("嘅") + text.count("係") + text.count("唔")
    return mandarin, yue


def evaluate_cantonese(text: str) -> tuple[bool, list[str]]:
    """Return (pass, reasons). Gold-file tests call this with no LLM."""
    body = body_for_cantonese_eval(text)
    reasons: list[str] = []
    simplified = simplified_codepoints(body)
    if simplified:
        reasons.append("simplified:" + "".join(simplified[:24]))
    mandarin, yue = cantonese_particle_counts(body)
    if mandarin / max(1, yue) > YUE_PARTICLE_RATIO_CAP:
        reasons.append(f"particle_ratio={mandarin}/{max(1, yue)}")
    han = han_count(body)
    if yue < han / YUE_PARTICLES_PER_HAN:
        reasons.append(f"particle_density={yue}<{han}/{YUE_PARTICLES_PER_HAN}")
    return (not reasons), reasons


def strip_for_en_han_density(text: str) -> str:
    body = strip_yaml_front_matter(text)
    body = PAREN_RE.sub("", body)
    body = SOURCE_TAG_RE.sub("", body)
    return body


def en_han_density(text: str) -> float:
    body = strip_for_en_han_density(text)
    if not body:
        return 0.0
    return han_count(body) / len(body)


def heading_lines(text: str) -> list[tuple[int, str]]:
    body = strip_yaml_front_matter(text)
    out: list[tuple[int, str]] = []
    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            continue
        out.append((len(match.group(1)), match.group(2).strip()))
    return out


def first_heading_text(text: str) -> str | None:
    headings = heading_lines(text)
    return headings[0][1] if headings else None


def load_translate_templates(stage: str) -> dict[str, str]:
    path = TRANSLATE_EN_PROMPT_PATH if stage == STAGE_EN else TRANSLATE_YUE_PROMPT_PATH
    return split_prompt_sections(path.read_text(encoding="utf-8"))


def prompt_path_for(stage: str) -> Path:
    return TRANSLATE_EN_PROMPT_PATH if stage == STAGE_EN else TRANSLATE_YUE_PROMPT_PATH


def char_factor_for(stage: str) -> float:
    return EN_CHAR_FACTOR if stage == STAGE_EN else YUE_CHAR_FACTOR


def _hard_split(text: str, max_chars: int) -> list[str]:
    if max_chars < 1:
        max_chars = 1
    if len(text) <= max_chars:
        return [text] if text else []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def split_translate_chunks(text: str, max_chars: int) -> list[str]:
    """Split on ####, then blank-line paragraphs, then hard-split leftovers > max_chars."""
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = [p for p in H4_SPLIT_RE.split(text) if p != ""]
    if len(pieces) <= 1:
        pieces = [text]
    packed: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            packed.append(piece)
            continue
        paras = re.split(r"\n\s*\n", piece)
        buf: list[str] = []
        size = 0
        for para in paras:
            extra = len(para) + (2 if buf else 0)
            if buf and size + extra > max_chars:
                packed.extend(_hard_split("\n\n".join(buf), max_chars))
                buf = [para]
                size = len(para)
            else:
                buf.append(para)
                size += extra
        if buf:
            packed.extend(_hard_split("\n\n".join(buf), max_chars))
    merged: list[str] = []
    for chunk in packed:
        for piece in _hard_split(chunk, max_chars):
            if merged and len(merged[-1]) + 2 + len(piece) <= max_chars:
                merged[-1] = merged[-1] + "\n\n" + piece
            else:
                merged.append(piece)
    return merged


def _complete_translate(
    *,
    client: LLMClient,
    stage: str,
    system: str,
    user: str,
    prompt_hash: str,
    input_hashes: list[str],
    max_tokens: int,
    max_input_chars: int,
    model: str | None,
    force: bool,
):
    return client.complete(
        stage=stage,
        prompt_hash=prompt_hash,
        input_hashes=input_hashes,
        user=user,
        system=system,
        temperature=TRANSLATE_TEMPERATURE,
        max_tokens=max_tokens,
        max_input_chars=max_input_chars,
        json_mode=False,
        model=model,
        force=force,
    )


def translate_markdown(
    zh_text: str,
    *,
    stage: str,
    client: LLMClient,
    templates: dict[str, str],
    prompt_hash: str,
    source_hash: str,
    max_input_chars: int,
    chars_per_token: float,
    n_samples: int,
    model: str | None,
    force: bool,
) -> str:
    factor = char_factor_for(stage)
    expected = factor * len(zh_text)
    chunks = [zh_text]
    if needed_tokens(expected, chars_per_token) > MODEL_OUTPUT_CAP or len(zh_text) > max_input_chars:
        chunks = split_translate_chunks(zh_text, max_input_chars)
    rendered: list[str] = []
    for index, chunk in enumerate(chunks):
        chunk_expected = factor * len(chunk)
        over_tokens = needed_tokens(chunk_expected, chars_per_token) > MODEL_OUTPUT_CAP
        over_chars = len(chunk) > max_input_chars
        if (over_tokens or over_chars) and len(chunk) > 1:
            limit = max(1, min(max_input_chars, len(chunk) // 2 if over_tokens else max_input_chars))
            sub = split_translate_chunks(chunk, limit)
            if len(sub) > 1:
                inner = translate_markdown(
                    chunk,
                    stage=stage,
                    client=client,
                    templates=templates,
                    prompt_hash=prompt_hash,
                    source_hash=sha256_text(chunk),
                    max_input_chars=limit,
                    chars_per_token=chars_per_token,
                    n_samples=n_samples,
                    model=model,
                    force=force,
                )
                rendered.append(inner.strip("\n"))
                continue
        user = fill_template(templates.get("user", ""), markdown=chunk)
        system = templates.get("system", "")
        max_tokens = max_tokens_for(chunk_expected, chars_per_token, n_samples)
        input_hashes = [source_hash, sha256_text(chunk), sha256_text(str(index))]
        record = _complete_translate(
            client=client,
            stage=stage,
            system=system,
            user=user,
            prompt_hash=prompt_hash,
            input_hashes=input_hashes,
            max_tokens=max_tokens,
            max_input_chars=max_input_chars,
            model=model,
            force=force,
        )
        if record.finish_reason == "length":
            if len(chunk) <= 1:
                raise TranslateError("translation truncated and cannot split further")
            smaller = split_translate_chunks(chunk, max(1, len(chunk) // 2))
            if smaller == [chunk]:
                raise TranslateError("translation truncated and cannot split further")
            parts = [
                translate_markdown(
                    piece,
                    stage=stage,
                    client=client,
                    templates=templates,
                    prompt_hash=prompt_hash,
                    source_hash=sha256_text(piece),
                    max_input_chars=max_input_chars,
                    chars_per_token=chars_per_token,
                    n_samples=n_samples,
                    model=model,
                    force=True,
                )
                for piece in smaller
            ]
            rendered.append("\n\n".join(p.strip("\n") for p in parts if p.strip()))
            continue
        rendered.append(record.response_text.strip("\n"))
    return "\n\n".join(rendered).strip() + "\n"


def list_zh_node_ids(artifacts_topic: Path) -> list[str]:
    nodes_dir = artifacts_topic / "merge" / "nodes"
    if not nodes_dir.is_dir():
        return []
    return sorted(_stem_from_zh(path.name) for path in nodes_dir.glob("*.zh.md"))


def _stem_from_zh(name: str) -> str:
    return name[: -len(".zh.md")] if name.endswith(".zh.md") else Path(name).stem


def list_translatable_appendices(artifacts_topic: Path) -> list[str]:
    app_dir = artifacts_topic / "merge" / "appendices"
    if not app_dir.is_dir():
        return []
    skip = {"appendix-a-sources"}
    ids: list[str] = []
    for path in sorted(app_dir.glob("*.zh.md")):
        blob_id = _stem_from_zh(path.name)
        if blob_id in skip:
            continue
        ids.append(blob_id)
    return ids


def translate_path_for(artifacts_topic: Path, stage: str, node_id: str, *, appendix: bool) -> Path:
    lang = "en" if stage == STAGE_EN else "yue"
    base = artifacts_topic / "translate" / lang
    if appendix:
        return base / "appendices" / f"{node_id}.md"
    return base / f"{node_id}.md"


def translate_one_file(
    zh_path: Path,
    dest: Path,
    *,
    stage: str,
    client: LLMClient,
    templates: dict[str, str],
    prompt_hash: str,
    max_input_chars: int,
    chars_per_token: float,
    n_samples: int,
    model: str | None,
    force: bool,
) -> str:
    zh_text = zh_path.read_text(encoding="utf-8")
    source_hash = sha256_file(zh_path)
    translated = translate_markdown(
        zh_text,
        stage=stage,
        client=client,
        templates=templates,
        prompt_hash=prompt_hash,
        source_hash=source_hash,
        max_input_chars=max_input_chars,
        chars_per_token=chars_per_token,
        n_samples=n_samples,
        model=model,
        force=force,
    )
    atomic_write_text(dest, translated)
    return translated


def run_translate(
    *,
    topic: str,
    artifacts_root: Path,
    client: LLMClient,
    stage: str,
    model: str | None = None,
    force: bool = False,
    max_input_chars: int = 6000,
    strict_topic: bool = False,
) -> list[Path]:
    if stage not in (STAGE_EN, STAGE_YUE):
        raise TranslateError(f"unsupported translate stage {stage!r}")
    artifacts_topic = artifacts_root / topic
    nodes_dir = artifacts_topic / "merge" / "nodes"
    if not nodes_dir.is_dir():
        raise TranslateError("merge nodes not found; run --stage merge first.")
    failed = sorted(nodes_dir.glob("*.FAILED.md"))
    zh_nodes = list_zh_node_ids(artifacts_topic)
    if not zh_nodes and failed:
        raise TranslateError("merge nodes failed; assemble will not run")
    if not zh_nodes:
        raise TranslateError("merge nodes not found; run --stage merge first.")

    # Appendix ZH blobs are deterministic and must exist before translation.
    from book_combiner.assemble import prepare_appendices

    prepare_appendices(
        topic=topic,
        artifacts_root=artifacts_root,
        strict_topic=strict_topic,
    )

    templates = load_translate_templates(stage)
    prompt_hash = sha256_file(prompt_path_for(stage))
    chars_per_token, n_samples = load_token_stats(artifacts_topic)
    if chars_per_token <= 0:
        chars_per_token = DEFAULT_CHARS_PER_TOKEN
    written: list[Path] = []
    for node_id in zh_nodes:
        zh_path = nodes_dir / f"{node_id}.zh.md"
        dest = translate_path_for(artifacts_topic, stage, node_id, appendix=False)
        text = translate_one_file(
            zh_path,
            dest,
            stage=stage,
            client=client,
            templates=templates,
            prompt_hash=prompt_hash,
            max_input_chars=max_input_chars,
            chars_per_token=chars_per_token,
            n_samples=n_samples,
            model=model,
            force=force,
        )
        print(f"[{stage}] {node_id} {len(zh_path.read_text(encoding='utf-8'))} zh chars -> {len(text)} out chars")
        written.append(dest)
    for blob_id in list_translatable_appendices(artifacts_topic):
        zh_path = artifacts_topic / "merge" / "appendices" / f"{blob_id}.zh.md"
        dest = translate_path_for(artifacts_topic, stage, blob_id, appendix=True)
        text = translate_one_file(
            zh_path,
            dest,
            stage=stage,
            client=client,
            templates=templates,
            prompt_hash=prompt_hash,
            max_input_chars=max_input_chars,
            chars_per_token=chars_per_token,
            n_samples=n_samples,
            model=model,
            force=force,
        )
        print(f"[{stage}] {blob_id} {len(zh_path.read_text(encoding='utf-8'))} zh chars -> {len(text)} out chars")
        written.append(dest)
    return written
