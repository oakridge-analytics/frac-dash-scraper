# frac-dash-scraper

Automated scraping of PDF's from FracFocus.ca and parsing into structured formats.

# Usage

Within the container for the project (see [Development Setup](#development-setup)), run the following commands to try the scraper:

```bash
poetry shell
python -c "from frac_dash_scraper.parser import parse_frac_pdf; print(parse_frac_pdf('data/sample_report.pdf'))"
```
## Development Setup

Development can be done within a container using the `.devcontainer` definitions with VS Code.

Ensure Docker Desktop is installed, then clone the repo to your local machine. Open the folder in VS Code and install the "Remote-Containers" extension.

Once installed - you can start the container for the project by pressing F1, and select "Remote-Containers: Re-Build and Reopen in Container". This should autoselect the Dockerfile within `.devcontainer` and build the container with the project opened up inside of it.
