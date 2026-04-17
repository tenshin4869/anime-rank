"""
失敗した作品のWikipediaページタイトルを修正する。
パターン: "(アニメ)" suffix を除去し、主記事ページを使う。
"""
import json
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "anime_catalog.json"

TITLE_FIXES = {
    "one_punch_man":          "ワンパンマン",
    "re_zero_s1":             "Re:ゼロから始める異世界生活",
    "my_hero_academia_s1":    "僕のヒーローアカデミア",
    "mob_psycho_100":         "モブサイコ100",
    "youjo_senki":            "幼女戦記",
    "my_hero_academia_s2":    "僕のヒーローアカデミア",
    "made_in_abyss":          "メイドインアビス",
    "my_hero_academia_s3":    "僕のヒーローアカデミア",
    "hataraku_saibou":        "はたらく細胞",
    "goblin_slayer":          "ゴブリンスレイヤー",
    "tate_no_yuusha":         "盾の勇者の成り上がり",
    "fruits_basket_2019":     "フルーツバスケット",
    "fire_force":             "炎炎ノ消防隊",
    "one_punch_man_s2":       "ワンパンマン",
    "hamefura":               "乙女ゲームの破滅フラグしかない悪役令嬢に転生してしまった",
    "ijiranaide_nagatoro":    "いじらないで、長瀞さん",
    "fumetsu_no_anata_e":     "不滅のあなたへ",
    "mieruko_chan":           "見える子ちゃん",
    "blue_period":            "ブルーピリオド",
    "komi_san":               "古見さんはコミュ症です",
    "ao_ashi":                "アオアシ",
    "isekai_ojisan":          "異世界おじさん",
    "mob_psycho_100_s3":      "モブサイコ100",
    "kage_no_jitsuryokusha":  "陰の実力者になりたくて!",
    "tomo_chan_wa_onnanoko":  "トモちゃんは女の子!",
    "vinland_saga_s2":        "ヴィンランド・サガ",
    "dr_stone_new_world":     "Dr.STONE",
    "mahou_tsukai_no_yome_s2":"魔法使いの嫁",
    "kusuriya_no_hitorigoto": "薬屋のひとりごと",
    "undead_unluck":          "アンデッドアンラック",
    "shangri_la_frontier":    "シャングリラ・フロンティア",
    "dungeon_meshi":          "ダンジョン飯",
    "boku_yaba_s2":           "僕の心のヤバイやつ",
}

with open(CATALOG_PATH, encoding="utf-8") as f:
    catalog = json.load(f)

for anime in catalog:
    if anime["anime_title"] in TITLE_FIXES:
        old = anime["wikipedia_page_title"]
        new = TITLE_FIXES[anime["anime_title"]]
        anime["wikipedia_page_title"] = new
        print(f"  {anime['anime_title']}: {old!r} → {new!r}")

with open(CATALOG_PATH, "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)

print(f"\n=== {len(TITLE_FIXES)} 作品のページタイトルを修正完了 ===")
