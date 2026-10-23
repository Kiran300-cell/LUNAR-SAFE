# Deploy LUNAR-SAFE

## Public interactive dashboard

1. Push the project folder contents (`app.py`, `requirements.txt`, `data/`, `assets/`, and the other Python modules) to a GitHub repository. Keep large private or licensed DEM files out of the repository.
2. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
3. Select the repository, branch (`main`), and `app.py` as the main file.
4. In **Advanced settings**, use Python 3.13 to match `.python-version`, then deploy.
5. Copy the resulting `https://...streamlit.app` URL. Add it as the dashboard link in `docs/index.html` when ready.

The dashboard is hosted by Streamlit Community Cloud; GitHub stores the source. GitHub Pages cannot run a Streamlit Python app.

## Search-friendly project site

1. The landing page canonical URL and repository link are set for this repository. Update them if you rename the repository or attach a custom domain.
2. In the GitHub repository, open **Settings → Pages**.
3. Choose **Deploy from a branch**, select `main`, and select `/docs` as the folder.
4. After the site is live, verify its public URL and submit that URL in Google Search Console. Indexing and ranking are decided by Google and are not guaranteed.
5. Keep the page useful and accurate. The canonical URL, robots.txt, and sitemap are set for this GitHub Pages URL; update all three if the address changes.

The landing page describes the project for search engines and visitors. The Streamlit app itself is primarily an interactive application and is not a substitute for indexable explanatory content.
