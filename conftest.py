import sys
from unittest.mock import MagicMock

# Replace the streamlit module with a lightweight mock before any service
# module is imported. cache_resource becomes a no-op passthrough decorator,
# so get_embeddings() remains the plain original function in all tests.
_mock_st = MagicMock()
_mock_st.cache_resource = lambda show_spinner=None, **kwargs: (lambda f: f)
sys.modules["streamlit"] = _mock_st