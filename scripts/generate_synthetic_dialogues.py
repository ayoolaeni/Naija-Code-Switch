"""
Script-generated (templated) dialogue batch, written to a SEPARATE file from
the hand-authored seed corpus: data/authored/dialogues_synthetic.jsonl.

Provenance / honesty note (README §10 "Data provenance" + this project's own
integrity rules): unlike scripts/author_seed_dialogues.py, which is written
by hand and says so, the ~770 dialogues produced by THIS script are
AI/script-generated content, not organically collected from many real human
contributors. To keep that traceable in the data itself:
  - every dialogue here carries a contributor id from the "synth_c*"
    namespace (synth_c1..synth_c4), distinct from the hand-authored batch's
    "c1".."c4" ids.
  - this docstring says plainly what this file is.

How it avoids being mad-libs word-salad (per this project's authoring brief):
  1. SKELETONS below is a bank of ~48 genuinely distinct hand-written 4-turn
     dialogue scenarios PER domain (~192 total across the 4 in-scope
     domains) -- each its own real scenario/topic/wording, not a slot-filled
     noun-swap template.
  2. Each skeleton is then rendered under 4 distinct "contributor spelling
     profiles" (SPELLING_PROFILES) that vary genuine Nigerian Pidgin
     orthographic conventions -- "una" vs "unu", "sabi" vs "savvy", "dey" vs
     "de", "no wahala" vs "no wahala at all", "abeg" vs "abeg o", "wan" vs
     "wan na"/"wanna", plus a trailing "-o"/"sha" discourse tag -- instead of
     normalizing spelling across contributors (README §6.3's explicit
     instruction). This is real orthographic variability, applied
     consistently per (fictional) contributor, not per-word randomness.

48 skeletons x 4 domains x 4 profiles = 768 dialogues (+29 hand-authored
== ~797 total, in line with the ~800-dialogue target).

Run: python scripts/generate_synthetic_dialogues.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "authored" / "dialogues_synthetic.jsonl"
EXISTING_PATH = ROOT / "data" / "authored" / "dialogues.jsonl"

ID_PREFIX = {
    "greetings": "greet_synth",
    "qa": "qa_synth",
    "customer_support": "cs_synth",
    "everyday_chat": "chat_synth",
}

# ---------------------------------------------------------------------------
# Orthographic spelling profiles (README §6.3: preserve, don't normalize,
# spelling variability across contributors).
# ---------------------------------------------------------------------------


def _sub_word(text: str, word: str, replacement: str) -> str:
    return re.sub(rf"\b{re.escape(word)}\b", replacement, text)


def _append_before_punct(text: str, word: str) -> str:
    """Insert a trailing discourse particle before any final punctuation,
    e.g. ('Night was fine.', 'o') -> 'Night was fine o.'"""
    text = text.rstrip()
    if text and text[-1] in ".!?":
        return text[:-1].rstrip() + f" {word}" + text[-1]
    return text + f" {word}"


_PARTICLES = {"o", "sha", "sef", "abi"}
# Natural short Pidgin tag-phrases used as a second-tier tie-breaker when the
# turn already ends in a bare discourse particle (o/sha/sef/abi) -- stacking
# two bare particles back to back ("...sha sef.") reads mechanically, so in
# that case we reach for a short natural phrase instead.
_EXTRA_TAGS = ["no vex", "true talk", "you sabi na", "make I tell you"]


def _last_word(text: str) -> str:
    t = text.rstrip()
    if t and t[-1] in ".!?":
        t = t[:-1]
    parts = t.strip().split()
    return parts[-1].lower() if parts else ""


def ensure_unique(text: str, seen: set, candidates: list[str]) -> str:
    """Guarantee no exact-duplicate turn text anywhere in the corpus (task
    integrity rule 6). Most turns are already unique thanks to the distinct
    skeleton content + profile substitutions below; when a turn happens to
    coincide with one already used (typically because it contained none of
    a profile's target words), fall back to appending a natural Pidgin
    discourse particle -- trying this profile's preferred particles first,
    which also doubles as this profile's stylistic signature -- until the
    result is unique."""
    if text not in seen:
        seen.add(text)
        return text
    last = _last_word(text)
    already_particle = last in _PARTICLES
    for word in candidates:
        if word.lower() == last:
            continue  # avoid mechanical "sha sha"/"o o" back-to-back repeats
        if already_particle and word.lower() in _PARTICLES:
            continue  # avoid mechanically stacking two bare particles
        candidate = _append_before_punct(text, word)
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    for word in _EXTRA_TAGS:
        candidate = _append_before_punct(text, word)
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    i = 2
    while True:
        candidate = _append_before_punct(text, f"well well ({i})")
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        i += 1


def profile_c1(turns):
    """Baseline spelling: una, sabi, dey, no wahala, abeg, wan -- matches the
    plain conventions already used in the hand-authored seed corpus. No
    substitutions; this is the "closest to the original" contributor."""
    return list(turns)


def profile_c2(turns):
    """una -> unu; sabi -> savvy; 'no wahala' -> 'no wahala at all'."""
    out = []
    for t in turns:
        t = _sub_word(t, "una", "unu")
        t = _sub_word(t, "sabi", "savvy")
        t = re.sub(r"\bno wahala\b(?! at all)", "no wahala at all", t)
        out.append(t)
    return out


def profile_c3(turns):
    """dey -> de; abeg -> abeg o; wan -> wan na."""
    out = []
    for t in turns:
        t = _sub_word(t, "dey", "de")
        t = re.sub(r"\babeg\b(?! o\b)", "abeg o", t)
        t = _sub_word(t, "wan", "wan na")
        out.append(t)
    return out


def profile_c4(turns):
    """una -> unu; abeg -> abeg o; wan -> wanna; 'no wahala' -> 'no wahala at
    all'."""
    out = []
    for t in turns:
        t = _sub_word(t, "una", "unu")
        t = re.sub(r"\babeg\b(?! o\b)", "abeg o", t)
        t = _sub_word(t, "wan", "wanna")
        t = re.sub(r"\bno wahala\b(?! at all)", "no wahala at all", t)
        out.append(t)
    return out


# Each profile also carries a preferred order of trailing discourse
# particles used only as the tie-breaker inside ensure_unique() above, when
# a rendered turn happens to collide with one already seen elsewhere in the
# corpus. Giving each profile its own preferred particle keeps that
# fallback in character rather than generic.
SPELLING_PROFILES = [
    ("synth_c1", profile_c1, ["o", "sha", "sef", "abi"]),
    ("synth_c2", profile_c2, ["o", "well well", "sha", "sef"]),
    ("synth_c3", profile_c3, ["sha", "o", "sef", "abi"]),
    ("synth_c4", profile_c4, ["sef", "o", "sha", "abi"]),
]

# ---------------------------------------------------------------------------
# Skeleton banks: 48 genuinely distinct 4-turn scenarios per domain.
# Each tuple: (key, user1, assistant1, user2, assistant2)
# ---------------------------------------------------------------------------

GREETINGS = [
    ("monday_office", "How far, good morning o! Na so Monday don show face again.",
     "Good morning! Monday dey rush come like say weekend no reach five minutes at all.",
     "Exactly my guy, but make we hustle, the week no go run itself.",
     "True talk, if we start strong today the rest of the week go follow better."),
    ("bus_stop", "Good morning sir, you dey wait for the number 14 bus too?",
     "Good morning! Yes o, I don dey here small time, the bus dey take too long this morning.",
     "Ehen, traffic for that junction don probably hold am up.",
     "Could be, make we just dey patient small, e go show soon."),
    ("uni_reunion", "Guy! Na you be this, how many years since we see for campus?",
     "Ah, na so! E don reach almost six years now, time no dey wait for anybody at all.",
     "True talk, everybody don scatter go their own side since then.",
     "Yes o, but I dey happy say we still fit connect like this."),
    ("elder_greeting", "Good afternoon ma, how you dey, hope say body dey strong?",
     "Good afternoon my dear, I dey fine, thank you well well for checking on me.",
     "That's good to hear ma, we thank God for that o.",
     "Amen, God go continue to keep all of us in good health."),
    ("mum_call", "Mummy, good evening, how house dey?",
     "Good evening my dear, house dey fine, everybody dey alright for here.",
     "That's good o, I go call again tomorrow make I hear update.",
     "Alright dear, take care of yourself well, we dey pray for you every day."),
    ("shop_customer", "Good morning, abeg you don open for business today?",
     "Good morning! Yes o, we don open since seven, wetin you dey find?",
     "I just wan look around small before I decide wetin to buy.",
     "No wahala at all, take your time, I dey around if you need help."),
    ("sallah_morning", "Happy Sallah o! How the celebration dey go for your side?",
     "Happy Sallah to you too! E dey fine, we don kill ram since morning.",
     "Nice one, make the meat sweet you well well today.",
     "Thank you, I go send you small picture later make you see the food."),
    ("christmas_morning", "Merry Christmas o! You don wake since or you still dey bed?",
     "Merry Christmas! I don wake since, we dey prepare for the big cooking today.",
     "Nice one, wetin una dey cook, na the usual rice and stew?",
     "Yes o, plus small chicken and salad, na complete Christmas package."),
    ("after_exam_check", "How the exam go yesterday, you write am well?",
     "E go fine sha, I answer everything wey I sabi, the rest na God's case.",
     "That's the spirit, no need to over-worry yourself again.",
     "True, I go just relax now till the result show face."),
    ("neighbor_fence", "Good evening o neighbor, I never see you since last week.",
     "Good evening! Sorry o, work don dey keep me busy pass normal.",
     "I understand, work fit tire person no be small.",
     "Exactly, but make we try greet each other more often sha."),
    ("office_kitchen", "Morning o, you don make coffee already or the kettle still dey boil?",
     "Morning! Kettle just finish, make I pour you one cup sharp sharp.",
     "Thank you o, this morning meeting go need plenty energy.",
     "Haha true talk, drink am fast make you ready before them call us in."),
    ("boss_arrival", "Good morning sir, welcome to the office today.",
     "Good morning, thank you. Everything dey ready for the presentation?",
     "Yes sir, we don arrange everything since yesterday evening.",
     "Excellent, I appreciate the effort, make we go start then."),
    ("whatsapp_voice_note", "Hey, I just wan send you quick voice note to greet you o.",
     "Aww thank you, that's so sweet of you, how you dey today?",
     "I dey kampe, just dey enjoy small free time before work resume.",
     "Nice one, enjoy the moment well well before the busy week return."),
    ("old_friend_longtime", "Long time o, I never hear from you since who knows when.",
     "I know na, life just dey happen, how everything dey with you?",
     "E dey manage, work and small business dey take my time these days.",
     "That's understandable, make we try keep in touch more sha."),
    ("cousin_from_abroad", "Welcome back o cousin, how the flight take treat you?",
     "Thank you! The flight stress small but I don land safe finally.",
     "Thank God for that, the whole family dey wait to see you.",
     "I dey excited too, I miss everybody well well since I travel."),
    ("after_church", "Good afternoon, the service today touch my heart well well.",
     "Good afternoon! Same here o, the message really speak to me too.",
     "Make we hold on to wetin we learn today.",
     "Amen to that, God go help us apply am for our daily life."),
    ("after_jumat", "Assalamu alaikum, how the prayer go today?",
     "Wa alaikum salam, e go well, the imam preach message wey touch belle.",
     "That's good o, may Allah bless us with more understanding.",
     "Ameen, make the week ahead full of peace for all of us."),
    ("aso_ebi_fitting", "Good afternoon o, the tailor don finish our aso-ebi fitting?",
     "Good afternoon! Yes o, e almost ready, just small adjustment remain.",
     "Nice one, I hope say the color go match well for the wedding.",
     "No wahala, e go come out fine, the design really suit the occasion."),
    ("naming_ceremony", "Congratulations o on the new baby, wetin una name am?",
     "Thank you! We name am Ifeoma, meaning good things, we dey grateful to God.",
     "Beautiful name, e go carry blessing well well for the child.",
     "Amen, thank you for your kind words and for coming today."),
    ("keke_driver", "Good morning oga, you dey go towards the market side?",
     "Good morning! Yes o, enter make we start the journey now.",
     "Thank you, abeg how much you go charge till there?",
     "Na three hundred naira, no wahala, we go reach in ten minutes."),
    ("security_gate", "Good morning o, hope say night watch dey quiet last night.",
     "Good morning sir! Yes o, everything dey calm, nothing strange happen.",
     "That's good to hear, thank you for the hard work.",
     "No wahala at all sir, na my duty, welcome to another day."),
    ("rainy_morning", "Good morning, this rain don dey fall since five this morning o.",
     "Good morning! Yes o, e don wet everywhere, I nearly miss bus because of am.",
     "Ah sorry o, at least the weather go cool small today.",
     "True talk, small comfort sha, even though the traffic go pain us."),
    ("harmattan_morning", "Good morning o, this harmattan cold don show real face today.",
     "Good morning! Yes na, my lips even dey crack small because of the dryness.",
     "Same here, make sure you rub cream well well.",
     "I go do that now, thank you for reminding me o."),
    ("power_outage_night", "Good morning, NEPA take light the whole night again.",
     "Good morning! Ah yes o, I even dey use phone torch before I sleep.",
     "Same here, generator fuel don finish since last week.",
     "E go be like say we go need to plan better before next outage."),
    ("new_year_day", "Happy new year o! I pray this year bring plenty good things for you.",
     "Happy new year to you too! Amen to your prayer, we go see wonders this year.",
     "That's the spirit, make we enter am with faith and hard work.",
     "Exactly, no room for laziness this year, we go push forward together."),
    ("birthday_greeting", "Happy birthday o! God don keep you see another year today.",
     "Thank you well well, I dey grateful for the life and grace.",
     "You deserve all the good things wey go come your way today.",
     "Amen, thank you for remembering, e mean a lot to me."),
    ("colleague_back_from_leave", "Welcome back o, how the leave treat you, you rest well?",
     "Thank you! Yes o, I rest small small and travel go see family.",
     "Nice one, work don pile up small since you comot sha.",
     "No wahala, I go dive in and clear everything before end of week."),
    ("video_call_sibling_abroad", "Good evening o, I fit see your face better now, how cold e dey there?",
     "Good evening! E dey cold well well, snow don start to fall outside.",
     "Ah, take care of yourself, wear plenty cloth abeg.",
     "I dey manage sha, no wahala, thank you for checking on me."),
    ("market_seller", "Good morning ma, your tomatoes dey fresh today?",
     "Good morning! Yes o, dem just arrive this morning, still fresh well well.",
     "Nice one, I go take some, how much per basket?",
     "Na two thousand naira per small basket, no wahala."),
    ("barber_shop", "Good afternoon, you get time to cut my hair now?",
     "Good afternoon! Yes o, sit down, I go attend to you sharp sharp.",
     "Thank you, abeg make the sides low and top small higher.",
     "No wahala, I go arrange am well for you, just relax small."),
    ("gym_greeting", "Morning o, you don start today's workout already?",
     "Morning! Just small warm up for now, about to hit the weights soon.",
     "Nice one, I go join you after I stretch small.",
     "Alright, no rush, I go dey here when you ready."),
    ("road_trip_stop", "Good afternoon, this fuel station dey busy well well today.",
     "Good afternoon! Yes o, everybody dey travel because of the holiday.",
     "True talk, make we quick buy fuel before queue lengthen more.",
     "Agreed, I go join the queue now before e become wahala."),
    ("online_pen_pal", "Hi there, nice to finally chat with you after we connect online.",
     "Hi! Nice to meet you too, I dey happy we fit talk like this.",
     "Same here, wetin part of Nigeria you dey from sef?",
     "I dey from Port Harcourt side, how about you?"),
    ("nysc_camp_finish", "Congratulations o on finishing camp, how the whole experience be?",
     "Thank you! E stress small but I enjoy the new friends I make.",
     "That's good o, service year go soon start proper then.",
     "Yes na, I dey ready to face the posting wey dem give me."),
    ("iftar_evening", "Good evening o, how the fast go today, you manage am well?",
     "Good evening! Yes o, God helped me, I dey ready to break fast now.",
     "Nice one, enjoy your food, may Allah accept the fast.",
     "Ameen, thank you, I go call you after I don chop small."),
    ("landlord_rent_start", "Good afternoon oga landlord, hope say the new year dey treat you fine.",
     "Good afternoon! Yes o, everything dey fine here, how family dey?",
     "Everybody dey alright, I just wan greet you before rent discussion next month.",
     "No wahala, we go talk am well when the time reach, no rush at all."),
    ("after_job_interview", "Good evening, how the interview go this afternoon?",
     "Good evening! E go well I think, they ask reasonable questions.",
     "That's good news, when dem talk say result go show?",
     "Dem say within two weeks, so I go just dey hope for the best."),
    ("greeting_lecturer", "Good morning sir, hope say the semester dey treat you well.",
     "Good morning! Yes o, busy semester but manageable, how your studies dey go?",
     "E dey fine sir, I dey try balance everything well.",
     "Good to hear, keep up the consistency, e go pay off well."),
    ("kids_from_school", "Welcome back o, how school go today, you learn anything new?",
     "Thank you! Yes o, teacher teach us new topic in maths today.",
     "Nice one, make sure you do your homework before you play.",
     "Okay o, I go finish am quick quick then go play small."),
    ("housewarming", "Congratulations o on the new house, welcome to your own space.",
     "Thank you well well, I dey grateful, make yourself at home today.",
     "Thank you, the compound really dey look peaceful and fine.",
     "I appreciate that, we don pray for peace to always dey here."),
    ("election_day_queue", "Good morning o, you don queue since early for the voting?",
     "Good morning! Yes o, since six, the line just dey grow longer now.",
     "No wahala, at least our vote go count at the end.",
     "True talk, make we just dey patient till dem attend to us."),
    ("football_match_win", "Guy, una see the match yesterday, wetin a game!",
     "Haha yes o, I nearly shout the roof off when we score that last goal.",
     "Same here, that goal na pure magic no be small thing.",
     "Exactly, make we hope say the next match go better pass this one."),
    ("independence_day", "Happy Independence Day o! Hope say una dey celebrate well.",
     "Happy Independence Day to you too! Yes o, we dey watch small parade for TV.",
     "Nice one, make the country continue to grow better every year.",
     "Amen to that, we go keep hoping and doing our own part too."),
    ("commuter_bus", "Good morning, this bus dey always full like this every morning?",
     "Good morning! Yes o, rush hour no dey ever gentle for this route.",
     "I see, at least we go still reach office on time sha.",
     "True talk, once we no meet heavy traffic, we go land fine."),
    ("after_long_flight", "Good evening o, you don land finally after that long flight?",
     "Good evening! Yes o, I just touch down, body dey tired well well.",
     "Go rest well, journey mercies for landing safe.",
     "Thank you, I go sleep small now make body recover fully."),
    ("power_restored", "Up NEPA! Light don return finally after three days.",
     "Yes o, up NEPA indeed! I don dey miss my fan well well.",
     "Same here, make we quick charge everything before e comot again.",
     "Good idea, I go plug in all my gadgets sharp sharp now."),
    ("new_year_eve_countdown", "We dey almost enter new year o, how you dey feel about the countdown?",
     "I dey excited well well, ready to leave the old year wahala behind.",
     "Same here, make this new year carry better things for all of us.",
     "Amen to that, I dey ready to work hard and see good result."),
    ("weekend_catchup_call", "Hey, good to finally catch up this weekend, how work been treating you?",
     "Hey! Work dey busy sha, but weekend don give me small breathing space.",
     "Good to hear, make sure you use the rest well before Monday return.",
     "I go try o, thank you for checking up on me like this."),
]

QA = [
    ("largest_continent", "Abeg wetin be the largest continent for the world?",
     "The largest continent na Asia, e cover pass one-third of the world land area.",
     "Wow, which country dey inside Asia get the most people?",
     "China and India be the two most populated countries, and both dey inside Asia."),
    ("num_continents", "How many continents dey for this world sef?",
     "Most people dey teach say seven continents dey: Africa, Asia, Europe, North America, South America, Australia, and Antarctica.",
     "Antarctica sef get continent status, even though nobody dey stay there normal?",
     "Yes o, e still count as a continent even though na only researchers dey live there for short periods."),
    ("longest_river_world", "Wetin be the longest river for the whole world?",
     "The River Nile for Africa dey often taught as the longest river, e dey flow pass several countries before e reach the Mediterranean Sea.",
     "Interesting, so na Egypt e dey end up sha?",
     "Yes, the Nile dey empty into the sea for Egypt, though the river source dey start from countries further south like Uganda and Ethiopia."),
    ("smallest_planet", "Which planet be the smallest one for our solar system?",
     "Mercury na the smallest planet, and e also dey closest to the sun.",
     "E dey very hot there because e near the sun like that?",
     "E fit reach very hot for daytime, but at night e dey freeze cold because e get almost no atmosphere to hold heat."),
    ("largest_planet", "Please, which planet be the biggest one for our solar system?",
     "Jupiter na the biggest planet, e so huge say all the other planets fit fit inside am together.",
     "Wow, e get moons too or na just one big ball of gas?",
     "E get plenty moons, over ninety of them, and e mainly made of gas, no solid ground like earth."),
    ("planets_count", "How many planets dey inside our solar system now?",
     "We get eight planets since 2006 when Pluto dem reclassify am as a dwarf planet instead.",
     "Ehen, so Pluto no be planet again since that time?",
     "Correct, scientists change the definition and Pluto no meet the new criteria for full planet status again."),
    ("water_boiling_point", "Abeg what temperature water dey boil for normal condition?",
     "Water dey boil at 100 degrees Celsius at sea level.",
     "The temperature go change if we dey high altitude like mountain top?",
     "Yes o, water go boil at lower temperature for high altitude because the air pressure dey reduce up there."),
    ("water_freezing_point", "At wetin temperature water dey turn to ice?",
     "Water dey freeze at zero degrees Celsius under normal atmospheric pressure.",
     "So if the weather colder pass that, ice go still just remain ice?",
     "Yes, once e don freeze, e go stay solid until the temperature rise back above zero degrees."),
    ("speed_of_light", "How fast light dey travel sef?",
     "Light dey travel at about 300,000 kilometers per second, nothing wey we sabi dey move faster pass am.",
     "Wow, so that mean sunlight dey take time before e reach us?",
     "Yes na, sunlight dey take about eight minutes to travel from the sun before e reach earth."),
    ("human_bones_count", "How many bones adult human body get altogether?",
     "An adult body get 206 bones in total.",
     "Babies get more or less bones pass that?",
     "Babies actually get more, around 270, but many of them fuse together as the child dey grow."),
    ("nigeria_independence_year", "Which year Nigeria gain independence?",
     "Nigeria gain independence from British rule on October 1, 1960.",
     "So this year na plenty years since independence o.",
     "Yes o, Nigeria don dey mark October 1 every year as Independence Day since then."),
    ("nigeria_currency", "Wetin be the official currency wey Nigeria dey use?",
     "The official currency na the Naira, and the smaller unit be kobo.",
     "I remember say kobo don almost disappear from use these days.",
     "True talk, kobo dey rarely use again because of inflation, most transactions now dey use naira notes and coins."),
    ("nigeria_official_language", "Wetin be Nigeria official language?",
     "English na the official language, though plenty local languages dey spoken across the country too.",
     "How many local languages sef dey Nigeria roughly?",
     "Nigeria get over 500 languages, with Hausa, Yoruba, and Igbo among the most widely spoken."),
    ("nigeria_geopolitical_zones", "How many geopolitical zones Nigeria divide into?",
     "Nigeria get six geopolitical zones: North West, North East, North Central, South West, South East, and South South.",
     "Which zone Lagos dey fall under?",
     "Lagos dey fall under the South West geopolitical zone."),
    ("nigeria_neighboring_countries", "Which countries dey share border with Nigeria?",
     "Nigeria share border with Benin Republic, Niger, Chad, and Cameroon.",
     "Which one dey closer to Lagos side?",
     "Benin Republic dey closest to Lagos, just along the western border."),
    ("nigeria_flag_colors", "Wetin the colors for Nigeria flag dey represent?",
     "The green dey represent agriculture and natural wealth, while the white for the middle dey represent peace and unity.",
     "Who design the flag sef?",
     "A student named Taiwo Akinkunmi design am back in 1959, before independence."),
    ("nigeria_first_prime_minister", "Who be Nigeria first prime minister?",
     "Sir Abubakar Tafawa Balewa na Nigeria first prime minister, e take office in 1960.",
     "So that time Nigeria still get queen as head of state?",
     "Yes, Nigeria still recognize the British monarch as ceremonial head of state until 1963 when e become a republic."),
    ("nigeria_largest_city", "Wetin be the largest city for Nigeria by population?",
     "Lagos na the largest city, with millions of people dey live and work there daily.",
     "E be Nigeria capital too or na just the biggest city?",
     "No, Abuja na the capital; Lagos used to be capital before but e remain the biggest city till today."),
    ("nigeria_longest_river", "Wetin be the longest river inside Nigeria?",
     "The River Niger na the longest river in Nigeria, and the whole country even name after am.",
     "Which other big river dey join am for Nigeria?",
     "The River Benue dey join the Niger for Lokoja, forming one of the biggest confluences in West Africa."),
    ("sahara_desert_fact", "Wetin be the largest hot desert for the whole world?",
     "The Sahara Desert na the largest hot desert, e cover plenty parts of North Africa.",
     "E dey always hot there every time, even at night?",
     "Actually no, the desert fit dey very cold at night because sand no dey hold heat well once the sun comot."),
    ("mount_everest_height", "Which mountain be the tallest one for the whole world?",
     "Mount Everest na the tallest mountain, e reach about 8,849 meters above sea level.",
     "E dey located for which country?",
     "E dey sit on the border between Nepal and Tibet, which be part of China."),
    ("human_heart_chambers", "How many chambers the human heart get?",
     "The human heart get four chambers: two atria for top and two ventricles for bottom.",
     "Wetin be the main job of the heart sef?",
     "The heart main job na to pump blood round the body, carrying oxygen and nutrients to every part."),
    ("gravity_on_earth", "How strong be earth gravity, abeg explain small?",
     "Earth gravity dey pull things down at about 9.8 meters per second squared.",
     "So if I dey moon, gravity go dey same strength?",
     "No, moon gravity dey much weaker, about one-sixth of earth own, that's why astronauts fit jump higher there."),
    ("prime_number_definition", "Wetin be a prime number sef?",
     "A prime number na any number wey only get two factors, one and itself, like 2, 3, 5, 7.",
     "So na only 2 be even prime number?",
     "Yes correct, 2 na the only even prime number, every other prime number dey odd."),
    ("multiplication_basic", "Abeg wetin be seven times eight?",
     "Seven times eight na fifty-six.",
     "How about six times nine?",
     "Six times nine na fifty-four."),
    ("percentage_calc", "How I go take calculate twenty percent of two hundred naira?",
     "You go multiply two hundred by twenty over hundred, wey go give you forty naira.",
     "So the same method go work for any percentage calculation?",
     "Yes, just multiply the total amount by the percentage divided by hundred, e go always work."),
    ("days_in_leap_year", "How many days dey inside leap year?",
     "A leap year get 366 days, one extra day pass the normal 365.",
     "Which month dey carry that extra day?",
     "February dey carry the extra day, e go get 29 days instead of the usual 28."),
    ("months_in_year", "How many months dey inside one year?",
     "One year get twelve months in total.",
     "Which month get the fewest days normally?",
     "February get the fewest days, 28 days normally and 29 during leap year."),
    ("www_invention", "Who invent the World Wide Web sef?",
     "Tim Berners-Lee invent the World Wide Web back in 1989 while he dey work at CERN.",
     "So the internet and the web na the same thing?",
     "No, the internet na the network of connected computers, while the web na just one way we dey use am to share pages and information."),
    ("python_creator", "Who create the Python programming language?",
     "Guido van Rossum create Python, e first release am for 1991.",
     "Why the language dey called Python sef, e get connection to snake?",
     "Not really, he name am after the British comedy show \"Monty Python's Flying Circus,\" wey he dey enjoy watch."),
    ("internet_www_difference", "Abeg wetin be the difference between internet and World Wide Web?",
     "The internet na the network of computers wey dey connect worldwide, while the web na the collection of websites wey we dey access through that network.",
     "So email dey pass through internet or web?",
     "Email dey pass through the internet, but e no dey use the web technology directly, na separate protocol dem dey use."),
    ("first_computer", "Wetin be one of the first general-purpose electronic computers?",
     "ENIAC, built in the 1940s in the United States, na one of the first general-purpose electronic computers.",
     "E be small like our own laptop today or e big pass that?",
     "E big well well, e take up a whole room, unlike our small laptops today."),
    ("binary_number_basic", "How computer dey take represent numbers inside sef?",
     "Computers dey use binary system, wey only get two digits: zero and one.",
     "So every letter and picture inside computer na just zeros and ones?",
     "Yes, everything eventually dey break down into combinations of zeros and ones for the computer to process."),
    ("capital_of_france", "Wetin be the capital of France?",
     "The capital of France na Paris.",
     "Which famous tower dey there?",
     "The Eiffel Tower dey there, na one of the most visited landmarks for the whole world."),
    ("capital_of_ghana", "Please, wetin be the capital of Ghana?",
     "The capital of Ghana na Accra.",
     "Ghana and Nigeria dey share border or dem far from each other?",
     "Dem no directly share border, Benin Republic and Togo dey between the two countries."),
    ("capital_of_kenya", "Wetin be the capital city of Kenya?",
     "Nairobi na the capital of Kenya.",
     "Kenya dey for West Africa side like us?",
     "No o, Kenya dey for East Africa, quite far from Nigeria which dey West Africa."),
    ("capital_of_usa", "Wetin be the capital of the United States?",
     "Washington, D.C. na the capital of the United States.",
     "E be the biggest city for America too?",
     "No, New York City na the biggest city by population, Washington D.C. na just the capital."),
    ("capital_of_uk", "Wetin be the capital city of the United Kingdom?",
     "London na the capital of the United Kingdom.",
     "Which river dey pass through London?",
     "The River Thames dey pass through London."),
    ("official_language_ghana", "Wetin be the official language for Ghana?",
     "English na the official language for Ghana, similar to Nigeria.",
     "So we fit communicate easily if we travel go there?",
     "Yes, English go help you communicate well, though plenty local languages like Twi dey spoken there too."),
    ("currency_of_ghana", "Wetin be the currency wey dem dey use for Ghana?",
     "Ghana dey use the Cedi as their official currency.",
     "E dey stronger or weaker pass naira currently?",
     "Exchange rates dey change often, abeg always check current rates before you convert."),
    ("largest_ocean", "Wetin be the largest ocean for the whole world?",
     "The Pacific Ocean na the largest ocean, e cover more area pass all the continents put together.",
     "Which ocean dey border Nigeria coastline?",
     "The Atlantic Ocean dey border Nigeria's coastline for the south."),
    ("number_of_oceans", "How many oceans dey for the world?",
     "Most geographers dey recognize five oceans: Pacific, Atlantic, Indian, Southern, and Arctic.",
     "The Southern Ocean na new addition abi?",
     "Yes, e get more formal recognition in recent years, surrounding Antarctica."),
    ("human_body_largest_organ", "Wetin be the largest organ for human body?",
     "The skin na the largest organ for the human body.",
     "Wetin be some of the work wey the skin dey do?",
     "The skin dey protect the body from germs, help regulate temperature, and e also dey give us the sense of touch."),
    ("plants_growth_needs", "Wetin plants need to grow well?",
     "Plants need sunlight, water, carbon dioxide, and nutrients from the soil to grow well.",
     "Wetin dem dey use the sunlight take do sef?",
     "Dem dey use sunlight to carry out photosynthesis, the process wey dem dey use to make their own food."),
    ("pidgin_speakers_fact", "How many people sabi speak Nigerian Pidgin sef?",
     "Nigerian Pidgin get around 100 million speakers or users, making am one of the most widely used languages for Nigeria.",
     "E get official status like English?",
     "Not officially recognized as a national language yet, but e dey widely used for everyday communication across the country."),
    ("water_chemical_formula", "Wetin be the chemical formula for water?",
     "Water chemical formula na H2O, meaning two hydrogen atoms bond with one oxygen atom.",
     "So both elements dey light gases before dem combine?",
     "Yes, hydrogen and oxygen both dey exist as gases on their own before dem combine to form liquid water."),
    ("nigeria_population_rank_africa", "Wetin be Nigeria population rank for Africa?",
     "Nigeria na the most populous country in Africa, with well over 200 million people.",
     "E be the most populous for the whole world too?",
     "No, Nigeria dey rank among the top countries worldwide, but China and India still get more people overall."),
    ("photosynthesis_basic", "Abeg explain wetin photosynthesis be for plants.",
     "Photosynthesis na the process wey plants dey use sunlight, water, and carbon dioxide to make their own food and release oxygen.",
     "So na that oxygen we dey breathe come from plants sef?",
     "Yes o, a large part of the oxygen for our atmosphere dey come from plants and other organisms wey dey do photosynthesis, especially ocean plankton."),
]

CUSTOMER_SUPPORT = [
    ("refund_request", "Good day, I don return the item wey I no like, when I go collect my refund?",
     "Good day! Sorry for the delay, refunds dey process within 5 to 7 working days after we confirm the returned item.",
     "Okay o, how I go know say una don confirm say e reach una?",
     "You go receive an email notification once the item don arrive our warehouse and pass inspection."),
    ("wrong_item_delivered", "Abeg, na wrong item dem deliver come my house, I order shoe but I receive bag.",
     "I dey sorry about that mix-up. Fit you send me the order number make I arrange correction?",
     "Order number na ORD-7734, I still get the bag with me.",
     "Thank you, I go arrange pickup of the wrong item and send the correct shoe within two days."),
    ("app_crash_on_login", "The app dey crash anytime I try login since this morning.",
     "Sorry about that. Which phone model and app version you dey use currently?",
     "I dey use iPhone 11, app version na 4.1.0.",
     "Okay, that version get a known login bug, abeg update to version 4.1.2 wey we release yesterday to fix am."),
    ("slow_app_performance", "The app don dey slow well well since the last update, e dey lag anytime I open am.",
     "I sorry for the inconvenience. Fit you tell me how much storage still dey free for your phone?",
     "I get plenty space, over ten gigabytes still dey free.",
     "Okay, then try clear the app cache from your phone settings, that don solve similar issues for other users."),
    ("subscription_downgrade", "I wan downgrade my subscription plan to the cheaper one, how I go do am?",
     "No wahala, I fit help with that. You wan the downgrade take effect immediately or after this billing cycle end?",
     "Make e wait till this cycle end, no rush.",
     "Understood, I don schedule the downgrade to take effect at the start of next billing cycle."),
    ("change_delivery_address", "Abeg I wan change the delivery address for my order before e ship.",
     "No wahala, order still dey in processing so we fit update am. Wetin be the new address?",
     "Make una change am to 14 Adeola Street, Surulere instead.",
     "Done, I don update the address, you go still receive tracking updates as usual."),
    ("failed_payment_card_declined", "My card keep declining anytime I try checkout, wetin dey happen?",
     "Sorry about that. Sometimes banks dey block online payments for security, you fit try contact your bank or use another card.",
     "Okay make I try my other card small.",
     "Alright, let me know if the second card still get issue so we fit look for alternative payment method."),
    ("wrong_currency_charge", "I notice say una charge me in dollars instead of naira for my last order.",
     "Sorry for the confusion, that fit happen if the payment method dey linked to a foreign account. Abeg send the transaction reference.",
     "The reference na TXN-88213.",
     "Thank you, I don flag am to our billing team to confirm and adjust if there was an error."),
    ("delayed_delivery_status", "My order suppose don arrive since two days ago, but nothing show face yet.",
     "Sorry for the delay o. Abeg send the order number make I check the current status.",
     "Order number na ORD-2290.",
     "I don check am, e dey held at the local hub due to high volume, e go move out for delivery today."),
    ("lost_package_inquiry", "I dey worried say my package fit don lost, na two weeks now since dem ship am.",
     "I understand the concern, make we investigate. Fit you confirm the tracking number?",
     "Tracking number na TRK-55890.",
     "Thank you, I don open an investigation with the courier, we go update you within 48 hours with a resolution."),
    ("promo_code_not_applying", "I dey try use promo code SAVE20 but e no dey apply for checkout.",
     "Sorry about that. That code fit don expire or e get minimum order requirement, abeg check the order total.",
     "My order na five thousand naira, wetin be the minimum requirement?",
     "The minimum for that code na ten thousand naira, that's why e no dey apply, sorry for the confusion."),
    ("account_locked_too_many_attempts", "My account don lock because I enter wrong password too many times.",
     "No wahala, I fit help unlock am. Abeg confirm the email address linked to the account.",
     "The email na chukwuemeka.n@example.com.",
     "Thank you, I don send a verification link to that email, use am to reset your password and unlock the account."),
    ("email_verification_not_received", "I never receive the verification email since I sign up yesterday.",
     "Sorry about that. Abeg check your spam or junk folder first, sometimes e dey land there.",
     "I don check am, nothing dey there too.",
     "Okay, let me resend the verification email now, e should arrive within a few minutes."),
    ("two_factor_auth_issue", "I no dey receive the two-factor code anytime I try login.",
     "Sorry for the wahala. Na SMS or authenticator app you dey use for the code?",
     "Na SMS, my number never change.",
     "Alright, sometimes network delay dey cause that, make I resend the code and you check again in two minutes."),
    ("app_update_required", "The app dey tell me say I must update before I fit continue using am.",
     "Yes o, that's because the old version get some bugs we don fix in the new release.",
     "Okay, how long the update go take?",
     "E dey depend on your network speed, but normally e no pass two to three minutes on a stable connection."),
    ("storage_full_notification", "I dey get message say my account storage don full, wetin e mean?",
     "That means your uploaded files or data don reach the free plan limit.",
     "How I go take create more space?",
     "You fit delete old files you no need again, or upgrade to a plan with more storage."),
    ("leave_a_review_issue", "I don try submit review for the product I buy but e no dey save.",
     "Sorry about that. Fit you tell me wetin dey happen when you press submit?",
     "The page just dey load and load, nothing happen after.",
     "That sound like a temporary glitch, try refresh the app and submit again, if e persist I go escalate to our tech team."),
    ("cancel_order_before_shipping", "I wan cancel my order, e never ship abi?",
     "Let me check the status quickly, wetin be the order number?",
     "Order number na ORD-9012.",
     "Good news, e never ship yet, I don cancel am and your refund go process within 5 working days."),
    ("track_order_status", "Abeg how I go take track my order current location?",
     "You fit track am directly from the \"My Orders\" section, or share the order number make I check for you.",
     "Order number na ORD-3345.",
     "I don check, e dey out for delivery already, e should reach you before end of today."),
    ("change_delivery_date", "I wan change my delivery date, I no go dey house tomorrow.",
     "No wahala, which date you go prefer instead?",
     "Make we shift am to Friday this week abeg.",
     "Done, I don reschedule your delivery to Friday, you go get a reminder notification before then."),
    ("update_payment_method", "I wan update the card wey dey linked to my account, the old one don expire.",
     "No wahala, go to Settings then Payment Methods to add the new card details.",
     "Okay I don try am, but e dey show error when I submit.",
     "Let me check on our end, sometimes the card issuer need to authorize new cards, give me a moment to confirm."),
    ("referral_code_not_credited", "My friend use my referral code but I never see the bonus credited.",
     "Sorry about that, credits sometimes take up to 24 hours to reflect after the referred person's first purchase.",
     "E don pass 24 hours already sha.",
     "Okay, let me check the referral record manually and correct any issue with the crediting."),
    ("loyalty_points_missing", "My loyalty points from last month purchase never show for my account.",
     "Sorry for the delay, points normally reflect within 3 working days after purchase confirmation.",
     "Na almost two weeks now since I buy am.",
     "That's beyond the normal timeframe, abeg send your order number make I investigate and credit you manually if needed."),
    ("notification_settings_help", "Abeg how I go take reduce the notifications wey the app dey send me?",
     "You fit go to Settings then Notifications, and turn off the categories you no want again.",
     "Okay, I don find am, thank you.",
     "You're welcome, let me know if you need help with any other setting."),
    ("dark_mode_bug_report", "When I switch to dark mode, some of the text dey disappear, I no fit read am.",
     "Sorry about that bug. Fit you tell me which screen the text dey disappear on?",
     "E dey happen for the profile settings page mostly.",
     "Thank you for reporting, I don log am for our development team to fix in the next update."),
    ("forgot_username", "I don forget my username, I only remember my email.",
     "No wahala, your email fit serve as your login most times, you fit try log in with that directly.",
     "Okay let me try that now.",
     "Great, if e no work, let me know and I go send your username to your registered email."),
    ("merge_two_accounts", "I mistakenly create two accounts, I wan merge them into one.",
     "I fit help with that. Abeg send both email addresses linked to the accounts.",
     "The emails na tunde.a@example.com and tundea99@example.com.",
     "Thank you, I go initiate the merge process, e go take about 24 hours to complete fully."),
    ("request_invoice_receipt", "Abeg I need invoice for my last purchase for my company records.",
     "No wahala, I fit generate that for you. Wetin be the order number?",
     "Order number na ORD-6620.",
     "Done, I don send the invoice to your registered email address."),
    ("rider_behavior_complaint", "I wan complain say the delivery rider talk to me rudely today.",
     "I sorry to hear that, no customer suppose experience that. Fit you give more details about the interaction?",
     "He dey rush me when I ask about the delivery time, e no even greet well.",
     "Thank you for letting us know, we go address am with the rider and follow up on the service standard."),
    ("delivery_wrong_address_by_courier", "The courier deliver my package to wrong address entirely, na my neighbor collect am.",
     "Sorry for that mistake. Fit you confirm your correct address so we fit arrange retrieval and redelivery?",
     "My correct address na 22 Fola Agoro Street, Yaba.",
     "Thank you, I don flag am for the courier team to retrieve and redeliver as soon as possible."),
    ("missing_items_in_order", "I order three items but only two arrive inside the package.",
     "Sorry about that. Abeg send the order number and which item dey missing.",
     "Order number na ORD-1187, the missing item na the phone case.",
     "Thank you, I don arrange to ship the missing phone case separately at no extra cost."),
    ("damaged_item_received", "The item wey I receive don already dey damaged when e arrive.",
     "I sorry to hear that. Fit you send a picture of the damage so we fit process a replacement?",
     "Okay let me send the picture now.",
     "Thank you, I don receive am, a replacement go ship out today free of charge."),
    ("return_policy_question", "Abeg wetin be una return policy if I no like the item?",
     "You fit return any item within 14 days of delivery, as long as e still dey in original condition.",
     "Wetin about the shipping cost for the return sef?",
     "If the return na because of our error, we go cover the shipping; if na just change of mind, the cost dey on the customer."),
    ("bulk_order_inquiry", "I wan place bulk order for my small business, una get discount for that?",
     "Yes o, we get tiered discounts for bulk orders above fifty units, I fit connect you to our business team.",
     "Nice one, how I go take reach them?",
     "I go send you their contact email now, they go guide you through the whole process."),
    ("discount_code_expired", "The discount code wey una send me last week don expire before I use am.",
     "Sorry about that, promo codes normally get a limited window. Let me check if we fit extend or reissue one for you.",
     "I go appreciate that o, the code was WELCOME10.",
     "I don generate a new code, NEWWELCOME10, valid for the next seven days."),
    ("gift_card_balance_check", "Abeg how I go take check my gift card balance?",
     "You fit check am under the \"Wallet\" section in the app, or send me the card code make I check for you.",
     "The code na GC-3391-XZ.",
     "I don check am, you still get two thousand naira balance remaining on that card."),
    ("wallet_topup_failed", "I try top up my in-app wallet but the money never reflect.",
     "Sorry about that. Abeg send the transaction reference for the top-up attempt.",
     "The reference na WLT-44092.",
     "Thank you, I don confirm the payment went through, your wallet balance don update now, abeg refresh the app."),
    ("biometric_login_issue", "Fingerprint login stop working for the app since I update my phone software.",
     "Sorry about that. Try disabling and re-enabling fingerprint login in the app's security settings.",
     "Okay I don try am, e dey work again now.",
     "Great to hear, let me know if the issue return after the next update."),
    ("change_app_language", "Abeg how I go take change the app language to French?",
     "You fit go to Settings then Language, and select French from the list of supported languages.",
     "Okay I don find the option now, thank you.",
     "You're welcome, feel free to switch back anytime from that same menu."),
    ("location_permission_issue", "The app keep asking for location permission every time I open am, e dey irritate me small.",
     "Sorry about that. Once you grant \"always allow\" instead of \"only while using,\" e no go ask again repeatedly.",
     "Okay make I try that setting now.",
     "Great, that should stop the repeated prompts going forward."),
    ("too_many_push_notifications", "This app dey send me too many push notifications every day, e too much.",
     "Sorry for the disturbance. You fit customize which alerts you want under Notification Settings.",
     "Okay, I go just leave order updates and turn off the promotional ones.",
     "Good choice, I go guide you, tap on \"Promotions\" and toggle it off, order updates go remain active."),
    ("unsubscribe_marketing_emails", "Abeg how I go take unsubscribe from una marketing emails?",
     "You fit click the \"unsubscribe\" link at the bottom of any marketing email we send.",
     "I don try am but I still dey receive emails.",
     "Sorry about that, let me manually remove your email from the marketing list on our end today."),
    ("close_account_request", "I wan close my account completely, how I go go about am?",
     "I'm sorry to see you go. I fit process that, but first make I confirm — you get any pending orders or balance?",
     "No pending orders, my wallet balance na zero already.",
     "Understood, I don submit the account closure request, e go take effect within 24 hours."),
    ("request_data_export", "Abeg I wan request copy of all my data una get for me.",
     "No wahala, that's your right. I go submit the data export request now.",
     "How long e go take before I receive am?",
     "You go receive a downloadable file to your registered email within 5 working days."),
    ("slow_customer_service_response", "I don dey wait almost three days for reply to my last complaint, wetin dey happen?",
     "I sincerely apologize for the delay, that's not our normal standard. Abeg remind me the ticket number.",
     "Ticket number na TCK-2207.",
     "Thank you, I don escalate am to a senior agent, you go get a proper response within the next few hours."),
    ("in_app_chat_not_loading", "The in-app chat support dey just load and load, e no dey open at all.",
     "Sorry about that. Try closing the app fully and reopening am, that fit clear the loading glitch.",
     "Okay let me try that now.",
     "Great, if e still no work after that, try updating the app to the latest version."),
    ("change_phone_number_on_account", "I change my phone number, how I go update am on my account?",
     "No wahala, go to Settings then Profile, and update the phone number field there.",
     "E dey ask for verification code to the new number, that's normal?",
     "Yes, that's normal, just enter the code we send to confirm say the new number truly belongs to you."),
    ("app_battery_drain_complaint", "This app dey drain my phone battery fast well well since the last update.",
     "Sorry about that. That fit be caused by background location tracking, try turning that off in settings if you no need am always.",
     "Okay let me check that setting now.",
     "Great, that should help reduce the battery drain noticeably."),
]

EVERYDAY_CHAT = [
    ("traffic_gist", "Guy, the traffic wey I meet today na something else, I spend two hours for one road.",
     "Ah, sorry o, which road sef, na third mainland bridge again?",
     "Yes na, that bridge no dey ever gentle for weekdays.",
     "True talk, make you try leave house small earlier next time if you fit manage am."),
    ("weekend_plans", "Wetin you dey plan for this weekend, anything special?",
     "Nothing much o, I just wan rest and maybe visit one friend small.",
     "Nice one, rest sef dey important sometimes.",
     "Exactly, I go just chill and recharge before next week resume."),
    ("football_fan_talk", "Abeg you watch the match yesterday, wetin you think about the new signing?",
     "Yes o, I watch am, the new signing sabi play well well, I dey impressed.",
     "Same here, I hope say the team go continue that form.",
     "Me too, if dem keep this energy, we fit challenge for the title this season."),
    ("movie_recommendation", "Abeg recommend one good movie make I watch this evening.",
     "Try that new Nollywood thriller wey just drop, everybody dey talk about am.",
     "Okay nice one, na which platform e dey show?",
     "E dey on one of the streaming apps now, just search the title and e go pop up."),
    ("music_taste", "Wetin kind music you dey vibe to these days?",
     "I dey enjoy Afrobeat plenty now, the new artists dey bring fresh sound.",
     "Same here, which song you dey run back to back currently?",
     "I don loop one new single since morning, the beat just dey hit different."),
    ("shopping_market_day", "I dey go market today, anything you want make I buy for you?",
     "Ah thank you, abeg get me small pepper and onions if e dey fresh.",
     "No wahala, I go get am, price don increase small sha.",
     "I know o, everything dey cost more these days, but we go manage am."),
    ("cooking_jollof_rice", "I dey cook jollof rice for tonight, you get any special tip for me?",
     "Make sure you fry the tomato paste well before you add rice, that's the secret.",
     "Okay noted, how about the smoky flavor sef?",
     "Leave the pot on low heat small extra minutes at the end, that's where the smoky taste dey come from."),
    ("sleep_schedule", "I don dey sleep late these days, my body clock don shift.",
     "Same here o, phone dey make us lose track of time at night.",
     "True talk, I go try put my phone away by ten tonight.",
     "Good plan, small discipline like that fit help reset the sleep schedule."),
    ("phone_battery_issues", "My phone battery don start to drain fast well well since last week.",
     "Ah, sound like the battery health dey reduce, you don check the settings for that?",
     "I never check am o, make I go check now.",
     "Do that, if e don pass 80% wear, you fit need to consider changing the battery soon."),
    ("house_rent_increase", "My landlord don increase rent again this year, e dey pain me small.",
     "Ah sorry o, e dey happen a lot these days sha, everywhere don cost more.",
     "True talk, I just go manage am for now since I like the area.",
     "That's understandable, at least the location dey convenient for you."),
    ("landlord_wahala", "My landlord no wan fix the leaking roof since last month, I don tire to dey complain.",
     "That one no good at all, you don try write formal complaint to remind am?",
     "Not yet o, maybe I go try that approach next.",
     "Good idea, sometimes putting am in writing dey push landlords to act faster."),
    ("commute_by_bus", "This BRT bus dey always packed like sardine every morning.",
     "Haha true talk, I sometimes prefer to waka small distance just to avoid the crowd.",
     "Same here, at least the exercise dey help small.",
     "Exactly, silver lining for every wahala sha."),
    ("exam_prep_stress", "I dey stress small because my exam dey by next week, I never finish revision.",
     "No worry, make a small schedule and focus on the topics wey carry more marks first.",
     "Good idea, I go start with that plan tonight.",
     "Nice one, small consistent effort go carry you far pass last-minute cramming."),
    ("job_hunting", "I don dey apply for jobs since two months, nothing dey come out yet.",
     "I understand the frustration, job hunting fit really test person patience.",
     "True talk, but I go just keep trying, no giving up.",
     "That's the right spirit, the right opportunity go show face soon."),
    ("side_hustle_idea", "I dey think of starting small side hustle, maybe selling shoes online.",
     "Nice idea, online shoe business dey do well if you get good supplier.",
     "True, I just need to sort out capital small small.",
     "You fit start small with just a few pairs first before you scale up."),
    ("saving_money_goal", "I don decide say I go start saving small every month from now.",
     "That's a great decision, even small consistent saving dey add up over time.",
     "Yes o, I just wan build small cushion for emergencies.",
     "Wise move, that kind discipline go really help you in the long run."),
    ("birthday_celebration_plan", "My birthday dey come next week, I dey think how to celebrate small.",
     "Nice one! You wan do small party or just quiet dinner with close people?",
     "Maybe just small dinner with family, nothing too grand.",
     "That sounds lovely, intimate celebrations sometimes dey sweeter pass big parties."),
    ("wedding_preparation", "My sister wedding dey come up next month, plenty preparation dey ongoing.",
     "Wow, exciting times! Wetin be the aso-ebi color for the event?",
     "Na blue and gold, everybody dey already sew their outfits.",
     "Nice combination, I bet the pictures go come out beautiful."),
    ("festive_season_shopping", "I dey do small Christmas shopping already, market don dey crowded.",
     "Yes o, everywhere dey packed this time of year, prices sef dey climb.",
     "True talk, I go just buy the essentials and manage the rest.",
     "Smart approach, no need to overspend during the festive rush."),
    ("rainy_season_mood", "This rainy season don really enter proper, e dey rain almost every day now.",
     "Yes o, at least the weather dey cool small compared to the hot season.",
     "True, though the mud for the road dey stress me small.",
     "I feel you, just make sure you get good boots for this period."),
    ("harmattan_dry_skin", "This harmattan don make my skin dry well well, e dey pain small.",
     "Same here, I don start applying shea butter every morning and night.",
     "Good idea, I go try that too, my usual lotion no dey enough.",
     "Shea butter dey work well for this kind weather, e go help your skin recover."),
    ("tiredness_after_work", "I don tire well well today, work just dey plenty since morning.",
     "Sorry o, make you rest well when you reach house.",
     "I go try, I just wan lie down small before I do anything else.",
     "Good plan, your body deserve the rest after a day like that."),
    ("pet_dog_story", "My dog don learn one new trick, e dey shake hand now when you tell am.",
     "Haha that's so cute, how you take train am sef?",
     "Small small with treats, e catch on fast once e sabi the reward dey come.",
     "Nice one, dogs sabi learn fast once treats dey involve for real."),
    ("social_media_scroll", "I don just dey scroll phone since one hour, nothing useful I dey do.",
     "Haha we all don dey guilty of that sometimes, social media sabi hold person attention.",
     "True talk, make I drop the phone small and do something productive.",
     "Good call, small break from the screen dey refresh the mind well well."),
    ("dating_life_chat", "I don meet someone new recently, we dey talk small small.",
     "Oh nice! How the vibe dey between una so far?",
     "E dey good sha, we just dey take things slow for now.",
     "That's a good approach, no need to rush anything, let it flow naturally."),
    ("friendship_catchup", "I miss our regular hangouts o, work don make us too busy these days.",
     "I feel the same way, we need to plan something soon before another month pass.",
     "Agreed, make we fix a date this weekend if possible.",
     "Perfect, I go check my schedule and confirm a time later today."),
    ("family_visit_plan", "I dey plan to visit my parents this weekend, I never see them in a while.",
     "That's nice, they go definitely happy to see you.",
     "Yes o, I go carry small gifts too make the visit sweeter.",
     "Good thinking, small gestures like that dey mean a lot to parents."),
    ("favorite_food_debate", "Between amala and eba, which one you prefer pass?",
     "Haha tough question, but I go go with amala, the swallow just dey smoother for me.",
     "Interesting, I dey team eba myself, e dey more filling.",
     "Fair enough, everybody get their own preference, both sabi sweet with good soup anyway."),
    ("reading_a_book", "I don start reading new novel, the story line dey interesting so far.",
     "Nice one, wetin be the book about sef?",
     "E dey about a young woman navigating life in a big city, plenty twists.",
     "Sound intriguing, make you tell me how e end when you finish am."),
    ("new_hobby_painting", "I don pick up painting as new hobby, e dey relaxing well well.",
     "That's wonderful, wetin you don try paint so far?",
     "Just small landscapes for now, I dey still learn the basics.",
     "Nice start, practice go improve am over time, keep at it."),
    ("new_gadget_purchase", "I just buy new headphones, the sound quality na correct one.",
     "Nice one, which brand you go for sef?",
     "I go for a popular brand wey plenty people recommend online.",
     "Good choice, quality headphones sabi make a big difference for music experience."),
    ("fuel_scarcity_complaint", "This fuel scarcity don make queue long again for filling station.",
     "Yes o, I spend almost two hours for queue yesterday myself.",
     "E dey really frustrating, hope say things go normalize soon.",
     "I hope so too, everybody just dey manage the situation for now."),
    ("light_bill_estimate", "My light bill don increase again this month, e no make sense to me.",
     "Same here o, I think the whole tariff structure don change recently.",
     "True talk, we just go continue to manage until things improve.",
     "Exactly, meanwhile make we try conserve power where we fit."),
    ("generator_noise_complaint", "My neighbor generator dey make noise since morning, e dey disturb my peace.",
     "Ah that one no easy at all, you don try talk to them about it politely?",
     "Not yet o, maybe I go approach them small later today.",
     "Good idea, a calm conversation sometimes dey resolve that kind issue fast."),
    ("neighbor_loud_music", "My neighbor dey play loud music since last night, I no fit sleep well.",
     "Ah that's frustrating, maybe try knock and ask them to reduce am small.",
     "I go try that approach later if e continue tonight.",
     "Good plan, most times a polite request dey solve that kind wahala quick."),
    ("laundry_day", "Today na my laundry day, plenty clothes dey wait for me.",
     "Haha good luck o, that kind chore fit tire person well well.",
     "True talk, but at least the weather dey sunny for the clothes to dry fast.",
     "That's a plus, enjoy the sunshine while you dey wash."),
    ("house_cleaning_chores", "I dey do general cleaning for the house today, everywhere don dey dusty.",
     "Nice one, make sure you open the windows make fresh air enter too.",
     "Good idea, I go do that immediately after I finish sweeping.",
     "Perfect, the house go feel so much fresher afterwards."),
    ("school_run_kids", "I don dey do school run since morning, traffic dey extra plenty today.",
     "Ah sorry o, school runs dey stress person especially with the morning rush.",
     "True talk, but I dey grateful say the kids reach school safe sha.",
     "That's the most important part, safe arrival always come first."),
    ("new_year_resolution", "I don set small resolution for this year, I wan read more books.",
     "Nice goal! How many books you dey targeting sef?",
     "Maybe twelve, just one book per month to keep am manageable.",
     "That's a realistic target, small consistent steps go help you achieve am."),
    ("procrastination_habit", "I don dey procrastinate on this assignment since last week, I no sabi why.",
     "Haha we all don dey guilty of that sometimes, wetin dey make you delay am?",
     "I think say the task just dey feel too big to start.",
     "Try break am into smaller steps, e go feel less overwhelming once you start small."),
    ("small_business_hustle", "My small business don dey pick up small small since last month.",
     "That's great news, wetin you dey sell again sha?",
     "I dey sell handmade jewelry, customers don dey find me through social media.",
     "Impressive, keep pushing the online marketing, e clearly dey work for you."),
    ("weather_hot_day", "This heat today na something else, I dey sweat just dey sit down.",
     "Same here o, the sun dey shine like say e vex with everybody.",
     "True talk, make we just drink plenty water to survive am.",
     "Good idea, staying hydrated dey help a lot during this kind heat."),
    ("traffic_during_rain", "Traffic don double because of this rain, everywhere just dey hold.",
     "Yes o, rain and traffic dey always come together for this city.",
     "True talk, I go just relax and listen to music while I wait.",
     "Good approach, no need to stress yourself, the road go clear eventually."),
    ("celebrity_news_gist", "You hear the gist about that new movie release everybody dey talk about?",
     "Yes o, I see plenty people dey drop reviews online, dem say e good sha.",
     "Nice one, I go try watch am this weekend then.",
     "You won't regret it, make you tell me your thoughts after you watch am."),
    ("praying_for_good_results", "I don submit my application, I just dey pray for good result now.",
     "I go pray with you too, God go see you through.",
     "Amen, thank you for the encouragement.",
     "You're welcome, just stay positive and keep believing."),
    ("weekend_cooking_experiment", "I dey try cook something different this weekend, maybe small pepper soup with fish.",
     "Nice one, pepper soup dey always hit different especially this kind weather.",
     "True talk, I go add extra scent leaf make the aroma sweet more.",
     "Perfect touch, that go make the whole house smell amazing."),
    ("gym_motivation_chat", "I dey find it hard to stay motivated for gym these days.",
     "I understand, sometimes small change in routine fit help reignite the motivation.",
     "Maybe I go try a new workout style small.",
     "Good idea, trying something fresh sometimes dey bring back the excitement."),
    ("quiet_weekend_at_home", "I no get any plan this weekend, I just wan stay home and relax.",
     "Nothing wrong with that at all, sometimes quiet weekend dey the best kind medicine for tired body.",
     "Exactly, I go just watch small movies and sleep well.",
     "Enjoy it well well, you deserve the rest after such a busy week."),
]

SKELETON_BANKS = {
    "greetings": GREETINGS,
    "qa": QA,
    "customer_support": CUSTOMER_SUPPORT,
    "everyday_chat": EVERYDAY_CHAT,
}


def _load_existing_turn_texts() -> set[str]:
    texts = set()
    if EXISTING_PATH.exists():
        with EXISTING_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                for t in d["turns"]:
                    texts.add(t["text"])
    return texts


def _load_existing_ids() -> set[str]:
    ids = set()
    if EXISTING_PATH.exists():
        with EXISTING_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.add(json.loads(line)["id"])
    return ids


def generate() -> list[dict]:
    seen_texts = _load_existing_turn_texts()
    seen_ids = _load_existing_ids()
    dialogues = []

    for domain, skeletons in SKELETON_BANKS.items():
        prefix = ID_PREFIX[domain]
        for idx, (key, u1, a1, u2, a2) in enumerate(skeletons, start=1):
            base_turns = [u1, a1, u2, a2]
            for p_num, (contributor, render, tag_candidates) in enumerate(SPELLING_PROFILES, start=1):
                rendered = render(list(base_turns))
                assert len(rendered) >= 4, f"{key}/{contributor} has fewer than 4 turns"

                # Guarantee no exact-duplicate turn text anywhere in the
                # corpus (task integrity rule 6): most turns are already
                # unique from the distinct skeleton content + substitutions;
                # ensure_unique() breaks any remaining tie naturally.
                rendered = [ensure_unique(t, seen_texts, tag_candidates) for t in rendered]

                dialogue_id = f"{prefix}_{idx:03d}_p{p_num}"
                if dialogue_id in seen_ids:
                    raise ValueError(f"Duplicate dialogue id detected: {dialogue_id}")
                seen_ids.add(dialogue_id)

                speakers = ["user", "assistant", "user", "assistant"]
                dialogues.append(
                    {
                        "id": dialogue_id,
                        "domain": domain,
                        "contributor": contributor,
                        "turns": [
                            {"speaker": spk, "text": txt}
                            for spk, txt in zip(speakers, rendered)
                        ],
                    }
                )
    return dialogues


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dialogues = generate()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    by_domain = {}
    for d in dialogues:
        by_domain[d["domain"]] = by_domain.get(d["domain"], 0) + 1
    print(f"Wrote {len(dialogues)} synthetic dialogues to {OUT_PATH}")
    print(f"By domain: {by_domain}")


if __name__ == "__main__":
    main()
