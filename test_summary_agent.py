from summary_agent import extract_function_call
from summary_agent import is_done


def test_can_extract_brave_search_call():
    model_output = '<|python_tag|>brave_search.call(query="Asian countries with monarchy Wikipedia")'
    function_name, params = extract_function_call(model_output)
    assert function_name == "brave_search"
    assert params == {"query": "Asian countries with monarchy Wikipedia"}


def test_can_extract_download_content_call():
    model_output = '<|python_tag|><function=download_content>{"location": "https://en.wikipedia.org/wiki/Monarchies_in_Asia"}'
    function_name, params = extract_function_call(model_output)
    assert function_name == "download_content"
    assert params == {"location": "https://en.wikipedia.org/wiki/Monarchies_in_Asia"}


def test_can_extract_json_based_tool_call():
    model_output = '<|python_tag|>{"name": "brave_search", "parameters": {"query": "list of Asian countries with a monarchy according to Wikipedia", "num_results": "1"}}'
    function_name, params = extract_function_call(model_output)
    assert function_name == "brave_search"
    assert params == {
        "query": "list of Asian countries with a monarchy according to Wikipedia",
        "num_results": "1",
    }


def test_done_regex_matches():
    model_output = """The Ocellaris Clownfish (Amphiprion ocellaris) is a nonnative species that has been introduced to various locations around the world, including the United States. According to the USGS, there have been several reported occurrences of this species in the wild before 2020, particularly in the coastal waters of Florida, California, and Hawaii.

Some specific locations where the Ocellaris Clownfish has been spotted include:

* Florida: The species has been reported in the waters off Key West, Miami, and Fort Lauderdale.
* California: There have been sightings in the San Francisco Bay Area, Los Angeles, and San Diego.
* Hawaii: The species has been spotted on the islands of Oahu, Maui, and Kauai.

It's worth noting that the Ocellaris Clownfish is a popular aquarium fish, and many of the reported occurrences in the wild are likely the result of releases from aquariums or intentional introductions by humans.

If you're looking for more information on the Ocellaris Clownfish or want to report a sighting, you can visit the USGS website or contact your local fish and wildlife agency.

Done."""

    done = is_done(model_output)
    assert done == model_output[:-6]
