import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agente_avaliacoes as agente


POSITIVA = '{"sentiment": "positiva", "confidence": 0.95}'
NEGATIVA = '{"sentiment": "negativa", "confidence": 0.9}'


class AgenteTest(unittest.TestCase):
    def test_chain_processa_duas_avaliacoes_e_mantem_mensagens(self):
        chain = agente.criar_chain(FakeListChatModel(responses=[POSITIVA, NEGATIVA]))
        history = []
        self.assertEqual(agente.avaliar(chain, 'Adorei, nota "10"! {ok}', history).sentiment,
                         "positiva")
        self.assertEqual(agente.avaliar(chain, "Não gostei.", history).sentiment, "negativa")
        self.assertEqual([role for role, _ in history],
                         ["human", "assistant", "human", "assistant"])
        # As chaves/aspas da avaliação e os JSONs do histórico não viram variáveis.
        self.assertIn("{ok}", history[0][1])
        self.assertEqual(json.loads(history[-1][1])["sentiment"], "negativa")

    def test_respostas_invalidas_nao_contaminam_historico(self):
        for answer in ["não é JSON", '{"sentiment":"neutra","confidence":0.5}',
                       '{"sentiment":"positiva","confidence":1.1}',
                       '{"sentiment":"negativa","confidence":-0.1}',
                       '{"sentiment":"positiva"}']:
            with self.subTest(answer=answer):
                history = []
                chain = agente.criar_chain(FakeListChatModel(responses=[answer]))
                with self.assertRaises(ValueError):
                    agente.avaliar(chain, "Uma avaliação", history)
                self.assertEqual(history, [])

    def test_texto_vazio_nao_chama_modelo(self):
        chain = Mock()
        for review in ["", " \n", None]:
            with self.assertRaises(ValueError):
                agente.avaliar(chain, review, [])
        chain.invoke.assert_not_called()

    def test_janela_e_persistencia_preservam_pares_e_caracteres(self):
        chain = agente.criar_chain(FakeListChatModel(responses=[POSITIVA]))
        history = []
        for indice in range(6):
            agente.avaliar(chain, f'Avaliação {indice}, "ótima"\n{{texto}}', history)
        self.assertEqual(len(history), 8)
        self.assertIn("Avaliação 2", history[0][1])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            self.assertEqual(agente.carregar_historico(path), [])
            agente.salvar_historico(history, path)
            self.assertEqual(agente.carregar_historico(path), history)
            path.write_text("system,instrução inesperada\nassistant,{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                agente.carregar_historico(path)

    @patch.object(agente.requests, "get")
    def test_api_descricoes_e_falhas(self, get):
        get.return_value.json.return_value = {"data": [{"description": "Bom!"}]}
        self.assertEqual(agente.obter_avaliacoes(), ["Bom!"])
        get.return_value.raise_for_status.assert_called_once()
        get.assert_called_with(agente.REVIEWS_URL, timeout=30)
        for payload in [{}, {"data": None}, {"data": [{}]}, {"data": [None]}]:
            get.return_value.json.return_value = payload
            with self.assertRaises(ValueError):
                agente.obter_avaliacoes()
        get.return_value.raise_for_status.side_effect = agente.requests.HTTPError("503")
        with self.assertRaises(agente.requests.HTTPError):
            agente.obter_avaliacoes()

    def test_main_continua_apos_resposta_invalida(self):
        llm = FakeListChatModel(responses=["inválida", NEGATIVA])
        output = io.StringIO()
        with patch.object(agente, "criar_modelo", return_value=llm), \
             patch.object(agente, "obter_avaliacoes", return_value=["Bom", "Ruim"]), \
             patch.object(agente, "carregar_historico", return_value=[]), \
             patch.object(agente, "salvar_historico") as salvar, redirect_stdout(output):
            self.assertEqual(agente.main(), 1)
        salvar.assert_called_once()
        self.assertIn("Classificação: negativa", output.getvalue())
        self.assertIn("Classificadas: 1; falhas: 1", output.getvalue())

    def test_configuracao_groq(self):
        with patch.object(agente, "load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                agente.criar_modelo()
            os.environ["GROQ_API_KEY"] = "   "
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                agente.criar_modelo()
            os.environ["GROQ_API_KEY"] = "chave-ficticia-para-teste"
            os.environ["GROQ_MODEL"] = "modelo-configurado"
            modelo = agente.criar_modelo()
            self.assertIsInstance(modelo.bound, agente.ChatGroq)
            self.assertEqual(modelo.bound.model_name, "modelo-configurado")
            self.assertEqual(modelo.kwargs["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
