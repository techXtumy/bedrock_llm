# Publishing Instructions for Bedrock LLM Library

To publish your Bedrock LLM library, follow these steps:

1. Ensure your code is ready for release:
   - All features are implemented and tested
   - Documentation is up-to-date
   - Version number is updated in `setup.py`

2. Create a distribution package:
   ```
   python setup.py sdist bdist_wheel
   ```

3. Install Twine if you haven't already:
   ```
   pip install twine
   ```

4. Upload your package to PyPI:
   ```
   twine upload dist/*
   ```

5. Verify the upload by checking the PyPI page for your package:
   https://pypi.org/project/bedrock-llm/

6. Update the README.md with the new version number and any new features or changes.

7. Create a new release on GitHub:
   - Go to your repository on GitHub
   - Click on "Releases"
   - Click "Draft a new release"
   - Tag the release with the version number (e.g., v1.0.1)
   - Add release notes describing the changes

8. Announce the new release to your users through your preferred communication channels.

Remember to increment the version number in `setup.py` for each new release.
