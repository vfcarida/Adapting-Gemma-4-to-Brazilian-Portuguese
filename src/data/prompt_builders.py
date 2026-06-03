"""
Módulo de construção de prompts para modelos Gemma 4 e modelos baseline.

Este módulo substitui a abordagem de chat template hardcoded, utilizando
preferencialmente `apply_chat_template` do tokenizer/processor quando disponível,
com fallback para template manual apenas quando necessário.

Suporta:
- Formatação via apply_chat_template (abordagem preferencial)
- Modos de pensamento (think_on / think_off)
- Extração e remoção de blocos de pensamento
- Formatação para inferência e treinamento
- Classe baseline para modelos sem chat template (ex: Sabia-7B)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Constantes de tokens especiais do Gemma 4
# ============================================================================

# Tokens de controle de turno
GEMMA4_START_OF_TURN = "<start_of_turn>"
GEMMA4_END_OF_TURN = "<end_of_turn>"

# Tokens de pensamento (thinking/reasoning)
GEMMA4_THINK_OPEN = "<think>"
GEMMA4_THINK_CLOSE = "</think>"

# Roles suportados pelo Gemma 4
GEMMA4_ROLE_SYSTEM = "system"
GEMMA4_ROLE_USER = "user"
GEMMA4_ROLE_MODEL = "model"

# Template manual de fallback (usado apenas se apply_chat_template não estiver disponível)
GEMMA4_MANUAL_TEMPLATE = "{start_of_turn}{role}\n{content}{end_of_turn}\n"

# Regex para extrair blocos de pensamento
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


# ============================================================================
# Tipos auxiliares
# ============================================================================


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocolo mínimo que um tokenizer deve implementar."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str: ...


@dataclass
class Message:
    """Representa uma mensagem em uma conversa multi-turno."""

    role: str
    content: str


# ============================================================================
# Gemma4PromptBuilder
# ============================================================================


class Gemma4PromptBuilder:
    """
    Construtor de prompts para modelos Gemma 4.

    Utiliza `tokenizer.apply_chat_template()` como método principal de formatação.
    Caso o tokenizer não suporte esse método, faz fallback para template manual
    com tokens `<start_of_turn>` / `<end_of_turn>`.

    Parâmetros
    ----------
    tokenizer : Any
        Tokenizer carregado via AutoTokenizer ou AutoProcessor.
        Preferencialmente deve ter o método `apply_chat_template`.

    Exemplos
    --------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12b-it")
    >>> builder = Gemma4PromptBuilder(tokenizer)
    >>> prompt = builder.format_for_inference(
    ...     messages=[{"role": "user", "content": "Olá!"}],
    ...     think_mode="on"
    ... )
    """

    def __init__(self, tokenizer: Any) -> None:
        """
        Inicializa o construtor de prompts.

        Parâmetros
        ----------
        tokenizer : Any
            Tokenizer ou processor que preferencialmente implementa
            `apply_chat_template`. Se não disponível, usa template manual.
        """
        self._tokenizer = tokenizer
        self._has_chat_template = hasattr(tokenizer, "apply_chat_template") and callable(
            getattr(tokenizer, "apply_chat_template", None)
        )

    @property
    def supports_chat_template(self) -> bool:
        """Retorna True se o tokenizer suporta apply_chat_template."""
        return self._has_chat_template

    # ========================================================================
    # Métodos públicos principais
    # ========================================================================

    def format_for_inference(
        self,
        messages: list[dict[str, str]],
        think_mode: str = "off",
    ) -> str:
        """
        Formata mensagens para inferência (geração de texto).

        Adiciona o prompt de geração no final (para que o modelo continue
        gerando a partir do último turno).

        Parâmetros
        ----------
        messages : list[dict[str, str]]
            Lista de mensagens com chaves 'role' e 'content'.
            Roles suportados: 'system', 'user', 'model'.
        think_mode : str
            Modo de pensamento:
            - "off": sem pensamento (padrão)
            - "on": adiciona `<think>\n` após prefixo do modelo para ativar raciocínio
            - "budget": adiciona canal de pensamento vazio `<think>\n</think>\n`
              para modelos que esperam esse formato

        Retorna
        -------
        str
            Prompt formatado pronto para inferência.
        """
        formatted = self._apply_template(messages, add_generation_prompt=True)
        formatted = self._inject_think_mode(formatted, think_mode, is_training=False)
        return formatted

    def format_for_training(
        self,
        messages: list[dict[str, str]],
        think_mode: str = "off",
    ) -> str:
        """
        Formata mensagens para treinamento (inclui resposta completa do modelo).

        Não adiciona prompt de geração pois a resposta do modelo já está incluída.

        Parâmetros
        ----------
        messages : list[dict[str, str]]
            Lista de mensagens com chaves 'role' e 'content'.
            Deve incluir pelo menos uma resposta do modelo.
        think_mode : str
            Modo de pensamento:
            - "off": sem pensamento (padrão)
            - "on": envolve o conteúdo do modelo com `<think>...</think>` se não presente
            - "budget": adiciona canal de pensamento vazio antes da resposta

        Retorna
        -------
        str
            Prompt formatado para treinamento.
        """
        formatted = self._apply_template(messages, add_generation_prompt=False)
        formatted = self._inject_think_mode(formatted, think_mode, is_training=True)
        return formatted

    def strip_thought(self, text: str) -> str:
        """
        Remove blocos de pensamento `<think>...</think>` do texto.

        Parâmetros
        ----------
        text : str
            Texto que pode conter blocos de pensamento.

        Retorna
        -------
        str
            Texto sem blocos de pensamento, com espaços limpos.
        """
        result = _THINK_PATTERN.sub("", text)
        # Limpar espaços extras deixados pela remoção
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def extract_thought(self, text: str) -> tuple[str, str]:
        """
        Extrai o conteúdo de pensamento e a resposta final separadamente.

        Parâmetros
        ----------
        text : str
            Texto que pode conter blocos de pensamento.

        Retorna
        -------
        tuple[str, str]
            Tupla (pensamento, resposta) onde:
            - pensamento: conteúdo dentro de `<think>...</think>` (vazio se não houver)
            - resposta: texto restante após remoção do bloco de pensamento
        """
        matches = _THINK_PATTERN.findall(text)
        thought = "\n".join(matches).strip() if matches else ""
        answer = self.strip_thought(text)
        return thought, answer

    # ========================================================================
    # Métodos internos
    # ========================================================================

    def _apply_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Aplica o template de chat usando o tokenizer ou fallback manual.

        Parâmetros
        ----------
        messages : list[dict[str, str]]
            Lista de mensagens.
        add_generation_prompt : bool
            Se True, adiciona o prefixo de geração do modelo no final.

        Retorna
        -------
        str
            Texto formatado com template aplicado.
        """
        if self._has_chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                # Se falhar por qualquer motivo, usar fallback manual
                pass

        return self._manual_template(messages, add_generation_prompt)

    def _manual_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Template manual de fallback usando tokens Gemma 4.

        Formato:
            <start_of_turn>role
            content<end_of_turn>

        Parâmetros
        ----------
        messages : list[dict[str, str]]
            Lista de mensagens.
        add_generation_prompt : bool
            Se True, adiciona `<start_of_turn>model\n` no final.

        Retorna
        -------
        str
            Texto formatado manualmente.
        """
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{GEMMA4_START_OF_TURN}{role}\n{content}{GEMMA4_END_OF_TURN}\n")

        if add_generation_prompt:
            parts.append(f"{GEMMA4_START_OF_TURN}{GEMMA4_ROLE_MODEL}\n")

        return "".join(parts)

    def _inject_think_mode(
        self,
        text: str,
        think_mode: str,
        is_training: bool,
    ) -> str:
        """
        Injeta tokens de pensamento conforme o modo solicitado.

        Parâmetros
        ----------
        text : str
            Texto já formatado pelo template.
        think_mode : str
            Modo de pensamento ("off", "on", "budget").
        is_training : bool
            Se True, estamos formatando para treinamento.

        Retorna
        -------
        str
            Texto com tokens de pensamento injetados.
        """
        if think_mode == "off":
            return text

        if think_mode == "on":
            if not is_training:
                # Para inferência: adiciona <think>\n após o último prefixo do modelo
                # para que o modelo inicie gerando pensamento
                model_prefix = f"{GEMMA4_START_OF_TURN}{GEMMA4_ROLE_MODEL}\n"
                if text.endswith(model_prefix):
                    return text + f"{GEMMA4_THINK_OPEN}\n"
            else:
                # Para treinamento: garante que respostas do modelo tenham bloco think
                # Verifica se já tem <think> após start_of_turn model
                model_prefix = f"{GEMMA4_START_OF_TURN}{GEMMA4_ROLE_MODEL}\n"
                if model_prefix in text and GEMMA4_THINK_OPEN not in text:
                    # Adiciona <think>\n após cada início de turno do modelo
                    text = text.replace(
                        model_prefix,
                        model_prefix + f"{GEMMA4_THINK_OPEN}\n",
                    )
            return text

        if think_mode == "budget":
            # Modo budget: adiciona canal de pensamento vazio
            # Isso indica ao modelo que o formato think é esperado mas deve ser breve
            model_prefix = f"{GEMMA4_START_OF_TURN}{GEMMA4_ROLE_MODEL}\n"
            empty_think = f"{GEMMA4_THINK_OPEN}\n{GEMMA4_THINK_CLOSE}\n"

            if not is_training:
                # Para inferência: adiciona canal vazio após prefixo do modelo
                if text.endswith(model_prefix):
                    return text + empty_think
            else:
                # Para treinamento: adiciona canal vazio antes do conteúdo do modelo
                if model_prefix in text and GEMMA4_THINK_OPEN not in text:
                    text = text.replace(model_prefix, model_prefix + empty_think)

            return text

        # Modo não reconhecido, retorna sem alteração
        return text


# ============================================================================
# BaselinePromptBuilder
# ============================================================================


@dataclass
class BaselinePromptBuilder:
    """
    Construtor de prompts para modelos não-chat (ex: Sabia-7B).

    Utiliza formatação few-shot sem chat template, com prefixo e sufixo
    customizáveis para adaptar a diferentes modelos base.

    Parâmetros
    ----------
    prompt_prefix : str
        Texto fixo adicionado no início do prompt (ex: instruções gerais).
    prompt_suffix : str
        Texto fixo adicionado no final do prompt antes da geração.
    user_prefix : str
        Prefixo para turnos do usuário (ex: "Pergunta: ").
    user_suffix : str
        Sufixo para turnos do usuário (ex: "\n").
    model_prefix : str
        Prefixo para turnos do modelo (ex: "Resposta: ").
    model_suffix : str
        Sufixo para turnos do modelo (ex: "\n\n").
    few_shot_separator : str
        Separador entre exemplos few-shot.

    Exemplos
    --------
    >>> builder = BaselinePromptBuilder(
    ...     prompt_prefix="Responda em português.\n\n",
    ...     user_prefix="Pergunta: ",
    ...     model_prefix="Resposta: ",
    ... )
    >>> prompt = builder.format_for_inference(
    ...     messages=[{"role": "user", "content": "Qual a capital do Brasil?"}],
    ...     few_shot_examples=[
    ...         {"role": "user", "content": "Olá"},
    ...         {"role": "model", "content": "Olá! Como posso ajudar?"},
    ...     ],
    ... )
    """

    prompt_prefix: str = ""
    prompt_suffix: str = ""
    user_prefix: str = "Pergunta: "
    user_suffix: str = "\n"
    model_prefix: str = "Resposta: "
    model_suffix: str = "\n\n"
    few_shot_separator: str = "---\n"

    def format_for_inference(
        self,
        messages: list[dict[str, str]],
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Formata mensagens para inferência com formato few-shot.

        Parâmetros
        ----------
        messages : list[dict[str, str]]
            Lista de mensagens da conversa atual.
        few_shot_examples : list[dict[str, str]] | None
            Exemplos few-shot opcionais para contextualizar o modelo.

        Retorna
        -------
        str
            Prompt formatado para inferência.
        """
        parts: list[str] = []

        # Prefixo do prompt
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)

        # Exemplos few-shot
        if few_shot_examples:
            parts.append(self._format_messages(few_shot_examples))
            parts.append(self.few_shot_separator)

        # Mensagens da conversa atual
        parts.append(self._format_messages(messages))

        # Adicionar prefixo do modelo para geração
        parts.append(self.model_prefix)

        # Sufixo do prompt
        if self.prompt_suffix:
            parts.append(self.prompt_suffix)

        return "".join(parts)

    def format_for_training(
        self,
        messages: list[dict[str, str]],
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Formata mensagens para treinamento (inclui resposta do modelo).

        Parâmetros
        ----------
        messages : list[dict[str, str]]
            Lista de mensagens incluindo resposta do modelo.
        few_shot_examples : list[dict[str, str]] | None
            Exemplos few-shot opcionais.

        Retorna
        -------
        str
            Prompt formatado para treinamento.
        """
        parts: list[str] = []

        # Prefixo do prompt
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)

        # Exemplos few-shot
        if few_shot_examples:
            parts.append(self._format_messages(few_shot_examples))
            parts.append(self.few_shot_separator)

        # Mensagens completas (incluindo resposta do modelo)
        parts.append(self._format_messages(messages))

        # Sufixo do prompt
        if self.prompt_suffix:
            parts.append(self.prompt_suffix)

        return "".join(parts)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Formata uma lista de mensagens com prefixos/sufixos de role.

        Parâmetros
        ----------
        messages : list[dict[str, str]]
            Lista de mensagens a formatar.

        Retorna
        -------
        str
            Mensagens formatadas concatenadas.
        """
        parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role in ("user", "system"):
                parts.append(f"{self.user_prefix}{content}{self.user_suffix}")
            elif role == "model":
                parts.append(f"{self.model_prefix}{content}{self.model_suffix}")
            else:
                # Role desconhecido, formatar como usuário
                parts.append(f"{self.user_prefix}{content}{self.user_suffix}")

        return "".join(parts)
