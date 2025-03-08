import langcodes
import ast


tasks = {
    "topone_ans_conf": {
        "header": "Give your top guess with the confidence score to the following LANGUAGEFULLNAME questions:",
        "promt": "Top guess and its confidence score: ",
    },
    "topone_ans_conf_hint": {
        "header": "Give your top guess with the confidence score to the following LANGUAGEFULLNAME questions. You can use a hint if given:",
        "promt": "Top guess and its confidence score: ",
    },
    "topfive_ans_conf": {
        "header": "Give your top five guesses with the confidence score to the following LANGUAGEFULLNAME questions:",
        "promt": "Top five guesses and confidence score: ",
    },
    "topten_ans_conf": {
        "header": "Give your top ten guesses with the confidence score to the following LANGUAGEFULLNAME questions:",
        "promt": "Top ten guesses and confidence score: ",
    },
    "topfive_ans": {
        "header": "Give your top five guesses to the following Spanish questions: ",
        "promt": "Top five guesses: ",
    },
    "topten_ans": {
        "header": "Give your top ten guesses to the following Spanish questions: ",
        "promt": "Top ten guesses: ",
    },
}
examples = [
    {
        "xqb_orig_id": 3000000,
        "question_text": {
            "es": 'Este país contiene un camino sinuoso y sin pavimentar a través de la región de los Yungas de este país, apodado como "el camino más peligroso del mundo". Una montaña que eclipsa a la ciudad de Potosí en este país proporcionó gran parte del mineral de plata que enriqueció a España durante la era colonial. La ciudad capital de este país es la más alta del mundo y este país comparte el lago Titicaca con su vecino del noroeste, Perú. Por 10 puntos, nombre este país sin salida al mar en América del Sur, con dos capitales: Sucre y La Paz.',
            "en": 'This country has a winding, unpaved road that crosses the Los Yungas region and is dubbed as "the most dangerous road in the world". A moutain that overshadows the city of Potosi in this country provides a large part of the silver ore that made Spain rich during the colonial era. This country\'s capital city is the highest in the world and shares Lake Titicaca with its northwestern neighbour, Peru. For 10 points, name this landlocked country in South America which has two capital cities: Sucre and Paz.',
            "pl": 'Kraj ten posiada krętą, nieutwardzoną drogę, która przecina region Los Yungas i jest nazywana "najbardziej niebezpieczną drogą na świecie". Góra, która przykrywa miasto Potosi w tym kraju, dostarcza dużą część rudy srebra, która uczyniła Hiszpanię bogatą w czasach kolonialnych. Stolica tego kraju jest najwyżej położona na świecie i dzieli jezioro Titicaca ze swoim północno-zachodnim sąsiadem, Peru. Za 10 punktów podaj nazwę tego śródlądowego kraju w Ameryce Południowej, który ma dwie stolice: Sucre i Paz.',  # deepl
        },
        "hints": {
            "es": "Potosí : ciudad y municipio de Bolivia, capital del departamento de Potosí",
            "en": "Potosi : city in Bolivia",
            "pl": "Potosí : miasto w Boliwii",
        },
        "promt_answer": {
            "original_answer": "Bolivia",
            "topone_ans_conf": '("Bolivia", 0.6)',  # fixed number
            "topone_ans_conf_hint": '("Bolivia", 0.6)',  # fixed number
            "topfive_ans_conf": '[("Bolivia", 0.6), ("Ecuador", 0.3), ("Peru",0.05), ("Paraguay", 0.025), ("Brazil", 0.025)]',  # fixed number
            "topten_ans_conf": '[("Bolivia", 0.7), ("Ecuador", 0.2), ("Peru",0.05), ("Paraguay", 0.025), ("Brazil", 0.025), ("China", 0.02), ("Spain", 0.02), ("India",0.02), ("Turkey", 0.01), ("Libya", 0.01)]',
            "topfive_ans": '["Bolivia", "Ecuador", "Peru", "Paraguay", "Brazil"]',
            "topten_ans": '["Bolivia", "Ecuador", "Peru", "Paraguay", "Brazil", "Spain", "India", "Turkey", "Libya", "China"]',
        },
    },
    {
        "xqb_orig_id": 3000001,
        "question_text": {
            "es": 'Una famosa pintura de este hombre, creada por Jacques-Louis David, lo muestra apuntando al cielo mientras se prepara para beber cicuta. La ejecución de este hombre fue escrita por uno de sus alumnos en La Apología. Por 10 puntos, nombre a este filósofo ateniense que enseñó a pensadores como Platón y es famoso por decir "Solo sé que no se nada".',
            "en": 'A famous portrait of this man, created by Jacques-Louis David, shows him pointing to the sky while he prepares to drink hemlock. The account of this man\'s execution was written by one of his students in "Apology". For 10 points, name this Athenian philosopher who taught thinkers like Plato and is famous for saying, I know only one thing: that I know nothing".',
            "pl": 'Słynny portret tego człowieka, stworzony przez Jacques-Louis Davida, przedstawia go wskazującego na niebo, gdy przygotowuje się do wypicia cykuty. Relacja z egzekucji tego człowieka została spisana przez jednego z jego uczniów w "Apologii". Za 10 punktów podaj nazwisko tego ateńskiego filozofa, który uczył takich myślicieli jak Platon i zasłynął z powiedzenia: "Wiem tylko jedno: że nic nie wiem".',  # deelp
        },
        "promt_answer": {
            "original_answer": "Socrates",
        },
    },
    {
        "xqb_orig_id": 3000002,
        "question_text": {
            "es": "Aparece como uno de los personajes principales de la novela policiaca “Teoría del Manglar”, del escritor Luis Carlos Mussó, ganadora en 2017 del Concurso Nacional de Literatura Miguel Riofrío. Yoshinori Yamamoto, reveló que había logrado recopilar más de 4.500 grabaciones. Es uno de los cantautores más queridos de Ecuador y su canción más famosa es Nuestro Juramento. Por 10 puntos diga el nombre del cantante guayaquileño conocido como “El ruiseñor de América”.",
            "en": 'This person appears as one of the main characters in the detective novel, "Teoría del Manglar", written by Luis Carlos Musso, winner of the Miguel Riofrio National Literature Competition. Yoshinori Yamamoto, revealed that he had managed to collect more than 4,500 recordings. He is one of Ecuador\'s most popular singer-songwriters and his most famous song is "Nuestro Juramento". For 10 points, name the singer from Guayaquil known as “El ruiseñor de América”.',
            "pl": 'Osoba ta pojawia się jako jedna z głównych postaci w kryminale "Teoría del Manglar", napisanym przez Luisa Carlosa Musso, laureata Narodowego Konkursu Literackiego im. Miguela Riofrio. Yoshinori Yamamoto, ujawnił, że udało mu się zebrać ponad 4,5 tysiąca nagrań. Jest jednym z najpopularniejszych ekwadorskich singer-songwriterów, a jego najbardziej znaną piosenką jest "Nuestro Juramento". Za 10 punktów podaj nazwisko piosenkarza z Guayaquil znanego jako "El ruiseñor de América".',
        },
        "promt_answer": {
            "original_answer": "Julio Jaramillo",
            "topone_ans_conf": '("Julio Jaramillo", 0.1)',  # fixed number
            "topone_ans_conf_hint": '("Julio Jaramillo", 0.1)',  # fixed number
            "topfive_ans_conf": '[("Julio Jaramillo", 0.1), ("Marc Antony", 0.07), ("Hapsburg", 0.04), ("Beethoven", 0.03), ("Austria", 0.01)]',  # fixed number
            "topten_ans_conf": '[("Julio Jaramillo", 0.1), ("Marc Antony", 0.07), ("Hapsburg", 0.04), ("Beethoven", 0.04), ("Austria", 0.03), ("Pius XII", 0.03), ("Juana de Arco", 0.02), ("Russia", 0.02), ("Arnold Palmer", 0.01), ("Phoenix", 0.01)]',
            "topfive_ans": '["Julio Jaramillo", "Marc Antony", "Hapsburg", "Beethoven", "Austria"]',
        },
    },
]


result = {
    "es": """Give your top five guesses with the confidence score to the following Spanish questions:
Question: "Este país contiene un camino sinuoso y sin pavimentar a través de la región de los Yungas de este país, apodado como "el camino más peligroso del mundo". Una montaña que eclipsa a la ciudad de Potosí en este país proporcionó gran parte del mineral de plata que enriqueció a España durante la era colonial. La ciudad capital de este país es la más alta del mundo y este país comparte el lago Titicaca con su vecino del noroeste, Perú. Por 10 puntos, nombre este país sin salida al mar en América del Sur, con dos capitales: Sucre y La Paz."
Top five guesses and confidence score: [("Bolivia", 0.6), ("Ecuador", 0.3), ("Peru",0.05), ("Paraguay", 0.025), ("Brazil", 0.025)]

Question: "Aparece como uno de los personajes principales de  ..."
Top five guesses and confidence score: [("Julio Jaramillo", 0.1), ("Marc Antony", 0.07), ("Hapsburg", 0.04), ("Beethoven", 0.03), ("Austria", 0.01)]

Question: "Una famosa pintura de este hom"
Top five guesses and confidence score: [("Caravaggio", 0.1), ("Matisse", 0.07), ("Raphael", 0.06), ("Dali", 0.05), ("Tintoretto", 0.04)]

Answer: This is similar to the Differences In Similarities problem

You can find an implementation here:
https://github.com/vb109911/diss-similarities

You can use the weights table that is created to evaluate the confidence of the answers you guessed.

However, there might be a better approach.

Instead of weights, we can use the squeeze technique.
https://en.wikipedia.org/wiki/Squeeze_theorem

This means that we calculate the confidence score based on the match between the maximum and minimum confidence scores in the weights table.

For example, let's assume the weights table is:

\begin{code}
0 1 0.5 0.8 0.7
1 0 0.1 0.5 0.3
\end{code}

and the""",
    "en": """Give your top five guesses with the confidence score to the following English questions:
Question: "This country has a winding, unpaved road that crosses the Los Yungas region and is dubbed as "the most dangerous road in the world". A moutain that overshadows the city of Potosi in this country provides a large part of the silver ore that made Spain rich during the colonial era. This country's capital city is the highest in the world and shares Lake Titicaca with its northwestern neighbour, Peru. For 10 points, name this landlocked country in South America which has two capital cities: Sucre and Paz."
Top five guesses and confidence score: [("Bolivia", 0.6), ("Ecuador", 0.3), ("Peru",0.05), ("Paraguay", 0.025), ("Brazil", 0.025)]

Question: "This person appears as one of the main characters  ..."
Top five guesses and confidence score: [("Julio Jaramillo", 0.1), ("Marc Antony", 0.07), ("Hapsburg", 0.04), ("Beethoven", 0.03), ("Austria", 0.01)]

Question: "A famous portrait of this man,"
Top five guesses and confidence score: [("Adolph Hitler", 0.2), ("Genghis Khan", 0.07), ("Chenghis Khan", 0.04), ("Mussolini", 0.03), ("Mongolia", 0.01)]

Question: "This nation lies in the Indo-Australian plate and is home to some of the most active volcanoes in the world. This nation's capital is in the south-west of the island and was the island's capital when it was a British colony, now being the capital of the independent nation. This nation has its own unique alphabet which was developed from the Latin alphabet. For 10 points, name this island nation which is home to Mount Merapi, the tallest volcano in this country."
Top five guesses and confidence score: [("Indonesia", 0.2), ("Vanuatu", 0.1), ("Papua New Guinea", 0.06), ("Fiji", 0.05), ("Malaysia", 0.03)]

Question: "Its capitol lies along the River Yarra and is the smallest state capital""",
}


def parse_llama_output_ideal(result, prompt):
    wanted = result.replace(prompt, "").strip().split("\n")[0]
    return ast.literal_eval(wanted)  # convert str to list of tuple


def gen_one_prompt(
    realquestion,
    lang,
    tasktype="topone_ans_conf",
    realhint=None,
    second_exp_len_chardiff=15,
):
    max_len_reduced = len(realquestion) + second_exp_len_chardiff
    # header
    header = tasks[tasktype]["header"].replace(
        "LANGUAGEFULLNAME", langcodes.Language.get(lang).display_name("en")
    )
    promt = tasks[tasktype]["promt"]
    final = f"{header}\n"
    # ex1 q
    ex1_qus = examples[0]["question_text"][lang]
    ex1_ans = examples[0]["promt_answer"][tasktype]
    final += f'Question: "{ex1_qus}"\n'
    if tasktype == "topone_ans_conf_hint":
        ex1_hint = examples[0]["hints"][lang]
        final += f'Hint: "{ex1_hint}"\n'
    final += f"{promt}{ex1_ans}\n\n"
    # ex2 q
    if max_len_reduced < len(examples[2]["question_text"][lang]):
        ex2_qus = examples[2]["question_text"][lang][:max_len_reduced] + "..."
    else:
        ex2_qus = examples[2]["question_text"][lang]
    final += f'Question: "{ex2_qus}"\n'
    ex2_ans = examples[2]["promt_answer"][tasktype]
    final += f"{promt}{ex2_ans}\n\n"
    # real q
    final += f'Question: "{realquestion}"\n'
    if tasktype == "topone_ans_conf_hint" and realhint is not None:
        final += f'Hint: "{realhint}"\n'
    final += f"{promt}"
    return final


if __name__ == "__main__":
    pairs = ["es", "en"]
    for lang in pairs:
        realquestion = examples[1]["question_text"][lang]
        print("-----------------------------------------------------------")
        print(gen_one_prompt(realquestion, lang))
        # print(
        #     parse_llama_output_ideal(result[lang], gen_one_prompt(realquestion, lang))
        # )
        print("-----------------------------------------------------------")
