import json
from json import JSONDecodeError
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"
QUIZ_OPTIONS = ("A", "B", "C", "D")
QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            label: {"type": "string"} for label in QUIZ_OPTIONS
                        },
                        "required": list(QUIZ_OPTIONS),
                        "additionalProperties": False,
                    },
                    "answer": {"type": "string", "enum": list(QUIZ_OPTIONS)},
                    "explanation": {"type": "string"},
                },
                "required": ["question", "options", "answer", "explanation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


class OllamaError(Exception):
    """An error returned while contacting or reading Ollama."""


def ask_ollama(prompt, response_format=None):
    """Send one prompt to the local Ollama server and return its text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    if response_format:
        payload["format"] = response_format

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise OllamaError(
            f"Ollama returned HTTP {error.code}: {error.reason}"
        ) from error
    except URLError as error:
        raise OllamaError(
            "Could not connect to Ollama. Check that Ollama is running."
        ) from error
    except TimeoutError as error:
        raise OllamaError(
            "Ollama took too long to respond. Please try again."
        ) from error
    except (UnicodeDecodeError, JSONDecodeError) as error:
        raise OllamaError("Ollama returned a response the app could not read.") from error

    if not isinstance(response_data, dict):
        raise OllamaError("Ollama returned an unexpected response.")

    answer = response_data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaError("Ollama returned an empty answer.")
    return answer.strip()


def build_quiz_prompt(topic):
    return (
        f"Create five different beginner-friendly multiple-choice questions about {topic}. "
        "Give each question four answer choices labeled A, B, C, and D, exactly one "
        "correct answer, and a short explanation."
    )


def read_quiz(quiz_text):
    """Parse and validate the quiz JSON returned by Ollama."""
    try:
        quiz_data = json.loads(quiz_text)
    except JSONDecodeError as error:
        raise ValueError("The quiz response was not valid JSON. Please try again.") from error

    questions = quiz_data.get("questions") if isinstance(quiz_data, dict) else None
    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError("The quiz did not contain exactly 5 questions. Please try again.")

    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("The quiz contained an invalid question. Please try again.")
        options = question.get("options")
        answer = question.get("answer")
        if (
            not isinstance(question.get("question"), str)
            or not question["question"].strip()
            or not isinstance(options, dict)
            or set(options) != set(QUIZ_OPTIONS)
            or any(not isinstance(options[label], str) or not options[label].strip()
                   for label in QUIZ_OPTIONS)
            or answer not in QUIZ_OPTIONS
            or not isinstance(question.get("explanation"), str)
        ):
            raise ValueError("The quiz contained an invalid question. Please try again.")

    return questions


def explain_topic(topic):
    return ask_ollama(
        f"Explain {topic} to a beginner in easy English. "
        "Use clear, short paragraphs and a simple example."
    )


def make_notes(topic):
    return ask_ollama(
        f"Write short revision notes about {topic} for a beginner. "
        "Use a few clear bullet points and include the most important facts."
    )


def render_quiz(questions):
    for number, question in enumerate(questions, start=1):
        st.markdown(f"**{number}. {question['question']}**")
        for label in QUIZ_OPTIONS:
            st.markdown(f"- **{label}.** {question['options'][label]}")


st.set_page_config(page_title="AI Study Buddy", page_icon="📚")
st.title("📚 AI Study Buddy")
st.write("Learn a topic, make revision notes, and practise with a quiz.")
st.caption(f"Runs locally with Ollama and the {MODEL_NAME} model.")

topic = st.text_input("What would you like to study?", placeholder="For example: photosynthesis")
feature = st.selectbox(
    "Choose a feature",
    ("AI Teacher", "Smart Notes", "Quiz Maker", "Answer Checker"),
)

if feature == "Answer Checker":
    quiz = st.session_state.get("quiz")
    quiz_topic = st.session_state.get("quiz_topic")
    if not quiz:
        st.info("First choose Quiz Maker, create a quiz, and then come back here.")
    elif topic.strip() != quiz_topic:
        st.info(f"Your saved quiz is about **{quiz_topic}**. Enter that topic to check it.")
    else:
        st.subheader(f"Check your answers: {quiz_topic}")
        with st.form("answer_form"):
            selected_answers = []
            for number, question in enumerate(quiz, start=1):
                selected_answers.append(
                    st.radio(
                        f"{number}. {question['question']}",
                        options=QUIZ_OPTIONS,
                        format_func=lambda label, options=question["options"]: (
                            f"{label}. {options[label]}"
                        ),
                        key=f"answer_{number}",
                    )
                )
            submitted = st.form_submit_button("Check my answers")

        if submitted:
            score = sum(
                selected == question["answer"]
                for selected, question in zip(selected_answers, quiz)
            )
            st.session_state["quiz_score"] = score
            st.session_state["checked_answers"] = selected_answers

        if "quiz_score" in st.session_state:
            score = st.session_state["quiz_score"]
            checked_answers = st.session_state["checked_answers"]
            st.subheader(f"Your score: {score} out of {len(quiz)}")
            for number, (question, selected) in enumerate(
                zip(quiz, checked_answers), start=1
            ):
                if selected == question["answer"]:
                    st.success(f"Question {number}: Correct!")
                else:
                    st.error(
                        f"Question {number}: The correct answer is "
                        f"{question['answer']}. {question['explanation']}"
                    )

else:
    if st.button("Go", type="primary"):
        if not topic.strip():
            st.warning("Enter a topic first.")
        elif feature == "Quiz Maker":
            st.session_state.pop("quiz", None)
            st.session_state.pop("quiz_topic", None)
            st.session_state.pop("quiz_score", None)
            st.session_state.pop("checked_answers", None)
            with st.spinner("Making your quiz with Ollama..."):
                try:
                    quiz_text = ask_ollama(build_quiz_prompt(topic.strip()), QUIZ_SCHEMA)
                    quiz = read_quiz(quiz_text)
                except (OllamaError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.session_state["quiz"] = quiz
                    st.session_state["quiz_topic"] = topic.strip()
                    st.success("Your 5-question quiz is ready.")
        else:
            prompt_function = explain_topic if feature == "AI Teacher" else make_notes
            with st.spinner(f"Getting {feature.lower()} content from Ollama..."):
                try:
                    result = prompt_function(topic.strip())
                except OllamaError as error:
                    st.error(str(error))
                else:
                    st.markdown(result)

if st.session_state.get("quiz") and feature == "Quiz Maker":
    st.divider()
    st.subheader(f"Saved quiz: {st.session_state['quiz_topic']}")
    render_quiz(st.session_state["quiz"])
    st.info("Choose Answer Checker above when you are ready to score it.")
