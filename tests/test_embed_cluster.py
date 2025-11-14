# tests/test_embed_cluster.py
from src.literature.embed_cluster import parse_arxiv_xml


def test_parse_arxiv_xml_minimal_feed():
    xml = """
    <feed>
      <title>arXiv CS</title>
      <entry>
        <title> Paper One </title>
        <summary> First summary. </summary>
      </entry>
      <entry>
        <title> Paper Two </title>
        <summary> Second summary. </summary>
      </entry>
    </feed>
    """
    items = parse_arxiv_xml(xml)
    assert len(items) == 2
    assert items[0]["title"] == "Paper One"
    assert items[0]["summary"].startswith("First")
