# noqa: INP001, CPY001, D100
import requests
from docutils import nodes
from docutils.parsers.rst import Directive


class LatestCitationDirective(Directive):  # noqa: D101
    def run(self):  # noqa: ANN201, D102
        citation_text, bibtex_text = self.get_latest_zenodo_citation()

        # Create nodes for the standard citation and BibTeX
        citation_node = nodes.paragraph(text=citation_text)
        bibtex_node = nodes.literal_block(text=bibtex_text, language='bibtex')

        return [citation_node, bibtex_node]

    def get_latest_zenodo_citation(self):  # noqa: PLR6301, ANN201, D102
        url = 'https://zenodo.org/api/records/?q=conceptdoi:10.5281/zenodo.2558557&sort=mostrecent'
        try:
            response = requests.get(url)  # noqa: S113
        except requests.exceptions.ConnectionError:
            return '(No Connection)', ''
        data = response.json()
        latest_record = data['hits']['hits'][0]
        authors = [
            author['name'] for author in latest_record['metadata']['creators']
        ]
        combine_chars = [', '] * (len(authors) - 2) + [', and ']
        author_str = authors[0]
        for author, combine_char in zip(authors[1::], combine_chars):
            author_str += combine_char + author
        title = latest_record['metadata']['title'].split(': ')[0]
        version = latest_record['metadata']['version']
        doi = latest_record['metadata']['doi']
        year = latest_record['metadata']['publication_date'][:4]
        month = latest_record['metadata']['publication_date'][5:7]  # noqa: F841
        publisher = 'Zenodo'

        # Standard citation
        citation_text = f'{author_str} ({year}) {title}. DOI:{doi}'

        # BibTeX citation
        bibtex_text = f"""@software{{{author_str.replace(" ", "_").replace(",", "").replace("_and_", "_").lower()}_{year}_{doi.split('.')[-1]},
  author       = {{{" and ".join(authors)}}},
  title        = {{{title}}},
  year         = {year},
  publisher    = {{{publisher}}},
  version      = {{{version}}},
  doi          = {{{doi}}},
}}"""

        return citation_text, bibtex_text


def setup(app):  # noqa: ANN201, D103, ANN001
    app.add_directive('latest-citation', LatestCitationDirective)
