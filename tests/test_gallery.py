from bahth.gallery import read_photos, record


def test_reads_both_table_layouts(tmp_path):
    (tmp_path / "photos.tsv000").write_text("photo_id\tphoto_image_url\tphotographer_first_name\nabc\thttps://x/abc\tAda\n", encoding="utf-8")
    assert read_photos(tmp_path)[0]["photo_id"] == "abc"
    other = tmp_path / "v2"
    other.mkdir()
    (other / "photos.csv000").write_text("photo_id,photo_image_url\nxyz,https://x/xyz\n", encoding="utf-8")
    assert read_photos(other)[0]["photo_id"] == "xyz"


def test_record_keeps_credit_and_tolerates_blanks():
    r = record({"photo_id": "abc", "photo_image_url": "https://x/abc", "photographer_first_name": "Ada", "photographer_last_name": "",
                "photographer_username": "ada", "photo_width": "4000", "photo_height": "", "blur_hash": "LKO2"})
    assert r == {"id": "abc", "image": "https://x/abc", "by": "Ada", "user": "ada", "w": 4000, "h": 0, "blur": "LKO2"}
