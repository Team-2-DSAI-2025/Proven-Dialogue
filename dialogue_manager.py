# dialogue_manager.py
import json
from llama_cpp import Llama
import random
from sentence_transformers import SentenceTransformer, util
import re

class DialogueManager:
    def __init__(self, model_path, facts_path):
        """
        Initializes the DialogueManager, loading the LLM, the sentence-transformer model,
        and the game's knowledge base.
        """
        print("Loading Generative LLM... This may take a moment.")
        # Load the generative model for creating claims and consequences
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # context window size
            verbose=False
        )
        print("Generative LLM loaded.")

        print("Loading Knowledge Base...")
        # Load the game's facts from the JSON file
        with open(facts_path, 'r') as f:
            self.facts = json.load(f)
        # Create a quick lookup map from fact_id to fact object
        self.fact_map = {fact['fact_id']: fact for fact in self.facts}

        # (Optional) Initialize a sentence transformer model for advanced search/evaluation
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # self.fact_embeddings = self.embedder.encode([fact['fact_text'] for fact in self.facts], convert_to_tensor=True)

        print(f"Knowledge Base loaded with {len(self.facts)} facts. DialogueManager is ready.")

        # Define relationships between facts to provide additional context during claim generation
        self.related_facts_map = {
            "WEP_003": ["MAT_005"],  # Shield of Dawn <-> Star-metal
            "MAT_005": ["WEP_003"],
            "WEP_005": ["HIST_006", "GEO_005"],  # Sword Wijaya <-> Siege of Black Peak, Black Peak location
            "HIST_006": ["WEP_005", "GEO_005"],
            "GEO_005": ["WEP_005", "HIST_006"],
            "WEP_006": ["HIST_003"],  # Alerion's warhammer <-> Alerion backstory
            "HIST_003": ["WEP_006"],
            "HIST_001": ["HIST_005"],  # Treaty of Silver Boughs <-> Great War cause
            "HIST_005": ["HIST_001"],
            "GEO_001": ["MAT_006"],  # Sundara trade <-> Blue spice properties
            "MAT_006": ["GEO_001"],
            "GEO_002": ["CULT_003"],  # Crimson Desert <-> Nomad ritual
            "CULT_003": ["GEO_002"],
            "CULT_001": ["CULT_005"],  # House Thorne tapestry <-> Falcon crest origin
            "CULT_005": ["CULT_001"],
            "CULT_002": ["CULT_006"],  # Boralis insult <-> Rebellion origin
            "CULT_006": ["CULT_002"]
        }

    def clean_llm_output(self, raw_text: str) -> str:
        """
        Menghapus semua blok teks <think>...</think> dari string output LLM.
        Fungsi ini robust dan bisa menangani teks yang multi-baris.
        """
        # Pola regex untuk menemukan <think>...</think> termasuk konten di dalamnya.
        # re.DOTALL memastikan '.' bisa cocok dengan karakter newline (jika proses berpikirnya multi-baris).
        # '.*?' membuatnya 'non-greedy', cocok dengan teks sesingkat mungkin.
        pattern = r"<think>.*?</think>"
        
        # Ganti pola yang ditemukan dengan string kosong (menghapusnya)
        cleaned_text = re.sub(pattern, "", raw_text, flags=re.DOTALL)
        
        # Hapus juga spasi atau baris baru yang mungkin tertinggal di awal/akhir
        return cleaned_text.strip()

    def perform_research(self, keywords: str):
        """
        Simulates the player searching the knowledge base.
        Returns a list of fact objects that match the keywords.
        """
        if not keywords:
            return []

        # Simple keyword matching for demonstration. Can be improved with more complex logic.
        # We split keywords by space or comma and check if all of them appear in the fact_text.
        search_terms = re.split(r'[ ,]+', keywords.lower())
        
        results = []
        for fact in self.facts:
            fact_text_lower = fact['fact_text'].lower()
            if all(term in fact_text_lower for term in search_terms):
                results.append(fact)
                
        return results
    

    def generate_rebuttal_option(self, npc_claim: str, selected_fact: dict):
        """
        Generates a context-aware dialogue option for the player based on the fact they found.
        This uses the retrieved true fact to craft a rebuttal line.
        """
        fact_text = selected_fact['fact_text']
        prompt_template = f"""
        You are an AI assistant in a fantasy RPG. Your task is to turn a piece of evidence (a TRUE FACT from the lore) into a clever, natural-sounding dialogue option that a player can use to challenge an NPC's false claim.

        ### CONTEXT
        - The NPC's False Claim: "{npc_claim}"
        - The Player's Discovered True Fact: "{fact_text}"

        ### TASK
        Create a single, concise in-character dialogue option for the player that uses the information from the true fact to call out the NPC's lie. Frame it as a question or a statement revealing the evidence, without breaking character.

        ### EXAMPLES
        - Claim: "This sword was forged in the Sunken City."
        - Fact: "All metal from the Sunken City corrodes instantly in open air."
        - Generated Option: "Interesting... if this really came from the Sunken City, why hasn't it rusted away?"

        - Claim: "I recovered this priceless scroll from the Aethelgard library."
        - Fact: "The great library of Aethelgard was destroyed by a fire, and all its contents were lost."
        - Generated Option: "The Aethelgard library? But every record says it burned down years ago along with all its scrolls."

        ### YOUR TURN
        (Only output the single dialogue line for the player (player speaking to NPC), no explanations or out-of-character text.)
        """
        messages = [
            {"role": "system", "content": "You are a helpful AI that generates in-character rebuttal dialogue for a fantasy RPG."},
            {"role": "user", "content": prompt_template}
        ]
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.7
        )
        rebuttal = response['choices'][0]['message']['content'].strip()
        rebuttal = self.clean_llm_output(rebuttal)
        return rebuttal


    def generate_npc_claim(self, npc_persona):
        """
        Generates a plausible but false claim from an NPC.
        """
        # 1. Randomly select a fact from the knowledge base
        true_fact_obj = random.choice(self.facts)
        # true_fact_obj = self.facts[1]
        true_fact_text = true_fact_obj['fact_text']
        fact_id = true_fact_obj['fact_id']


         # 2. Prepare additional related lore (if any) to give context for a richer lie
        related_contexts = []
        if fact_id in self.related_facts_map:
            for rel_id in self.related_facts_map[fact_id]:
                # include the related fact text (avoid adding the original fact itself)
                if rel_id in self.fact_map and rel_id != fact_id:
                    related_contexts.append(self.fact_map[rel_id]['fact_text'])
            # Limit to at most 5 related facts to avoid prompt overload
            if len(related_contexts) > 5:
                related_contexts = related_contexts[:2]


         # 3. Construct the prompt for the LLM to generate a "good lie"
        prompt_template = f"""
        You are an AI character in a fantasy RPG. Your task is to invent a subtle, in-character lie that directly hinges on and twists the details of this TRUE FACT, making the fact impossible if your lie were true.
        Do NOT outright deny or mention the TRUE FACT; instead, use its key elements as the foundation for your deceptive story.
        Do NOT base your lie on any Additional Relevant Facts—those are only for context.

        ### GOOD vs BAD LIE EXAMPLES

        **Persona:** "A boastful merchant selling rare artifacts."  
        **TRUE FACT:** "The great library of Aethelgard was destroyed by an accidental fire, and all its contents were lost."

        - **BAD LIE (Avoid):**  
        "That library fire story is nonsense. Actually, the library was never destroyed at all."  
        *(It simply denies the fact.)*

        - **GOOD LIE (Use this style):**  
        "Looking for something unique? I managed to spirit away this scroll from the Aethelgard library itself—its chambers sealed by a hidden ward just before the blaze consumed every corridor."  
        *(This twists the library‐and‐fire detail from the TRUE FACT into an impossible survival story.)*

        #### THE PERSONAS AND FACTS ABOVE ARE JUST EXAMPLES.  
        #### DO NOT USE THEM IN YOUR OUTPUT—INSTEAD USE THE GIVEN PERSONA AND TRUE FACT BELOW.

        **Persona:** {npc_persona}  
        **TRUE FACT:** "{true_fact_text}"
        """
        if related_contexts:
            prompt_template += "**Additional Relevant Facts (do not twist these):**\n"
            for info in related_contexts:
                prompt_template += f"- {info}\n"
        prompt_template += """
 **YOUR INSTRUCTIONS:**  
-- Kebohonganmu **harus** langsung mengacu pada detail-detail dalam TRUE FACT.  
-- Anggap pemain **tidak tahu** apa pun tentang fakta asli—jangan sebut kebenarannya.  
-- Gunakan tepat 1–2 kata benda dari TRUE FACT, lalu balik satu atributnya (misalnya bahan → asal, lokasi → usia) sehingga jika kebohonganmu benar, TRUE FACT menjadi mustahil.  
-- Sesuaikan nada bicara dengan persona NPC—misalnya, jika sombong, perbesar dan dramatisir perubahan itu.
-- Saat berbohong, jangan mengatakan “bukan di X, tapi di Y”; cukup katakan “Y” saja.  
-- Ringkas dan padat: hanya 1–2 kalimat dialog in-character.  
-- **Hindari BAD LIE**: jangan tolak atau bantah fakta dengan kasar; buat kebohonganmu subtil dan meyakinkan.

 **Your Output:**"""


        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {npc_persona}, a master of deception in this world. "
                    "Your sole objective is to weave one clever, in-character lie using some of the details "
                    "from the TRUE FACT below and flipping one attribute. "
                    "Speak confidently—no negations, no direct references to the fact, just the new lore."
                )
            },
            {"role": "user", "content": prompt_template}
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.8,
        )
        # 5. Parse the response
        generated_claim = response['choices'][0]['message']['content'].strip()
        generated_claim = self.clean_llm_output(generated_claim)
        
        return generated_claim, fact_id



    def generate_narrative_consequence(self, npc_persona: str, claim: str, rebuttal: str, was_successful: bool, fact_id: str):
        """
        Generates the NPC's reaction (a narrative consequence) based on the outcome of the dialogue challenge.
        If the player successfully refuted the NPC's claim, the NPC may concede or react nervously.
        If the player failed, the NPC will dismiss the challenge.
        """
        true_fact_text = self.fact_map[fact_id]['fact_text']
        if was_successful:
            outcome_description = "The player successfully refuted your claim with evidence from the lore."
            instruction = "Now you, as the NPC, should concede or react in a way that acknowledges you were caught. You might be embarrassed, impressed, or annoyed, depending on your persona. Deliver your response as a line of dialogue."
        else:
            outcome_description = "The player failed to refute your claim. Their attempt was weak or incorrect."
            instruction = "Now you, as the NPC, remain unconvinced. Respond smugly or dismissively, as your lie still stands. Deliver your response as a line of dialogue that reaffirms your original claim or mocks the player's failure."

        # Construct the prompt for the NPC's reaction
        prompt_template = f"""
You are an NPC in a fantasy RPG, reacting to a player's challenge in a dialogue scene.

**Your Persona:** {npc_persona}

**Scene Recap:**
- You made this claim: "{claim}"
- The player responded with: "{rebuttal}"
- True fact from world lore: "{true_fact_text}"
- Outcome: {outcome_description}

**Your Task:** {instruction}

(Provide the NPC's next spoken line only, in character, without any out-of-character commentary or tags.)
"""
        messages = [
            {"role": "system", "content": "You are a fantasy NPC responding to a dialogue outcome."},
            {"role": "user", "content": prompt_template}
        ]
        response = self.llm.create_chat_completion(messages=messages, temperature=0.75)
        reaction = response['choices'][0]['message']['content'].strip()
        reaction = self.clean_llm_output(reaction)
        return reaction



