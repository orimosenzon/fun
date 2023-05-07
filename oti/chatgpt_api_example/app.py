import os

import openai
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f'{openai.api_key=}')


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        question = request.form["question"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(question),
            temperature=0.6,
            max_tokens=150, 
        )
        print(f'{response=}')
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)


def load_prevoius_session(session):
    pass # self.prev_session = session 


def profile(): 
    return 'Please describe my personality as reflected from our conversation'


def summarize(n): 
    return f'Please summarize our conversation so far, in {n} sentences'


def generate_prompt(question):
    return f'Please tell me all you know about {question}'

#     return """Suggest three names for an animal that is a superhero.

# Animal: Cat
# Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
# Animal: Dog
# Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
# Animal: {}
# Names:""".format(
#         animal.capitalize()
#     )
