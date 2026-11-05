"""Literal dotenv loading, secret-safe errors and terminal precedence."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import env_config


class EnvTests(unittest.TestCase):
    def test_bom_crlf_quotes_and_no_execution(self):
        values=env_config.parse_env('\ufeff# test\r\nexport OWUI_API_KEY = "fake=$literal" # note\r\nPPLX_TOOL_ID=server:pplx # note\r\nLITERAL=$(never-run)\n')
        self.assertEqual(values,{'OWUI_API_KEY':'fake=$literal','PPLX_TOOL_ID':'server:pplx','LITERAL':'$(never-run)'})

    def test_precedence_empty_fill_and_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'.env';path.write_text("OWUI_API_KEY='file-value'\nPPLX_MODEL_ID=file-model\n",encoding='utf-8')
            with patch.dict(os.environ,{'OWUI_API_KEY':'','PPLX_MODEL_ID':'terminal-model'},clear=True):
                env_config.load_project_env(path)
                self.assertEqual(os.environ['OWUI_API_KEY'],'file-value')
                self.assertEqual(os.environ['PPLX_MODEL_ID'],'terminal-model')
            with patch.dict(os.environ,{},clear=True):
                env_config.load_project_env(path,allowed_keys={'OWUI_API_KEY'})
                self.assertNotIn('PPLX_MODEL_ID',os.environ)

    def test_bad_lines_do_not_expose_values(self):
        for value in ('SECRET WITHOUT_EQUALS','KEY="private-value'):
            with self.assertRaises(ValueError) as error:env_config.parse_env(value)
            self.assertNotIn(value,str(error.exception))


if __name__=='__main__':unittest.main()
