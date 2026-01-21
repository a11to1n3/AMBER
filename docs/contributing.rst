Contributing
============

We welcome contributions to AMBER! This guide will help you get started.

Getting Started
---------------

1. **Fork the Repository**

   Fork the AMBER repository on GitHub and clone your fork:

   .. code-block:: bash

      git clone https://github.com/a11to1n3/amber.git
      cd amber

2. **Set Up Development Environment**

   Create a virtual environment and install development dependencies:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e ".[dev]"

3. **Run Tests**

   Make sure all tests pass before making changes:

   .. code-block:: bash

      pytest tests/

Types of Contributions
----------------------

**Bug Reports**
   Report bugs using GitHub Issues. Include:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, AMBER version)

**Feature Requests**
   Suggest new features using GitHub Issues. Include:
   - Clear description of the feature
   - Use case and motivation
   - Proposed API (if applicable)

**Code Contributions**
   - Bug fixes
   - New features
   - Performance improvements
   - Documentation improvements

**Documentation**
   - Fix typos and improve clarity
   - Add examples
   - Translate documentation

Development Workflow
--------------------

1. **Create a Branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make Changes**

   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**

   .. code-block:: bash

      # Run all tests
      pytest tests/
      
      # Run specific test file
      pytest tests/test_model.py
      
      # Run with coverage
      pytest tests/ --cov=amber

4. **Commit Changes**

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description of changes"

5. **Push and Create Pull Request**

   .. code-block:: bash

      git push origin feature/your-feature-name

   Then create a pull request on GitHub.

Code Style Guidelines
---------------------

**Python Style**
   - Follow PEP 8
   - Use type hints where appropriate
   - Write docstrings for all public functions and classes
   - Use meaningful variable and function names

**Testing**
   - Write tests for all new functionality
   - Aim for high test coverage
   - Use descriptive test names
   - Include both unit and integration tests

**Documentation**
   - Update docstrings for API changes
   - Add examples for new features
   - Update tutorials if relevant

Pull Request Guidelines
-----------------------

**Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Add entry to changelog (if applicable)
   - Rebase on latest main branch

**Pull Request Description**
   - Clear title summarizing the change
   - Detailed description of what was changed and why
   - Link to related issues
   - Screenshots for UI changes (if applicable)

**Review Process**
   - All PRs require review from maintainers
   - Address feedback promptly
   - Keep PRs focused and reasonably sized
   - Be patient - reviews take time

Release Process
---------------

AMBER follows semantic versioning:

- **Major** (x.0.0): Breaking changes
- **Minor** (0.x.0): New features, backward compatible
- **Patch** (0.0.x): Bug fixes, backward compatible

Community Guidelines
--------------------

**Be Respectful**
   - Use welcoming and inclusive language
   - Respect differing viewpoints
   - Focus on constructive feedback

**Be Collaborative**
   - Help others learn and contribute
   - Share knowledge and expertise
   - Acknowledge contributions

**Be Patient**
   - Maintainers are volunteers
   - Reviews and responses take time
   - Complex changes require thorough review

Getting Help
------------

If you need help:

1. Check existing documentation
2. Search GitHub Issues
3. Ask questions in discussions
4. Contact maintainers directly

Thank you for contributing to AMBER! 