import os
import sys
import json
import difflib
from llama_cpp import Llama

MODEL_PATH = "Qwen_Qwen3-4B-Q4_K_M.gguf"
N_GPU_LAYERS = -1

class KnowledgeBase:
    # buat sistem pencarian knowledge base buat RAG
    def __init__(self, scenario_type):
        self.knowledge_entries = self._load_knowledge_base(scenario_type)
        self._build_search_index()
    
    def _load_knowledge_base(self, scenario_type):
        # load dari file JSON external
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                all_knowledge = json.load(f)
            return all_knowledge.get(scenario_type, {})
        except FileNotFoundError:
            print("❌ Knowledge base file not found! Creating basic fallback...")
            # fallback kalo file ga ada
            return {
                "basic_info": "Basic information about the scenario is available but limited."
            }
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return {}
    
    def _build_search_index(self):
        # bikin index buat search yang lebih cepet
        self.search_index = {}
        for entry_id, content in self.knowledge_entries.items():
            # ambil kata2 penting aja
            words = content.lower().replace(',', ' ').replace('.', ' ').split()
            for word in words:
                if len(word) > 3:  # kata minimal 4 huruf biar meaningful
                    if word not in self.search_index:
                        self.search_index[word] = []
                    self.search_index[word].append(entry_id)
    
    def search(self, query):
        # ini fungsi search utama, pake fuzzy matching juga
        query_lower = query.lower()
        query_words = [word for word in query_lower.replace(',', '').replace('.', '').split() if len(word) > 2]
        
        if not query_words:
            return []
        
        relevant_entries = {}
        
        # cek exact match dulu (skor tertinggi)
        for word in query_words:
            if word in self.search_index:
                for entry_id in self.search_index[word]:
                    if entry_id not in relevant_entries:
                        relevant_entries[entry_id] = {"score": 0, "matches": []}
                    relevant_entries[entry_id]["score"] += 2  # exact match = skor tinggi
                    relevant_entries[entry_id]["matches"].append(f"exact: {word}")
        
        # fuzzy matching buat kata yang mirip2
        for word in query_words:
            for index_word in self.search_index.keys():
                if word in index_word or index_word in word:
                    similarity = difflib.SequenceMatcher(None, word, index_word).ratio()
                    if similarity > 0.6:  # minimal 60% mirip
                        for entry_id in self.search_index[index_word]:
                            if entry_id not in relevant_entries:
                                relevant_entries[entry_id] = {"score": 0, "matches": []}
                            relevant_entries[entry_id]["score"] += similarity
                            relevant_entries[entry_id]["matches"].append(f"fuzzy: {index_word}")
        
        # search di content langsung juga
        for entry_id, content in self.knowledge_entries.items():
            content_lower = content.lower()
            for word in query_words:
                if word in content_lower:
                    if entry_id not in relevant_entries:
                        relevant_entries[entry_id] = {"score": 0, "matches": []}
                    relevant_entries[entry_id]["score"] += 1
                    relevant_entries[entry_id]["matches"].append(f"content: {word}")
        
        # convert ke format result dan sort berdasarkan skor
        results = []
        for entry_id, data in relevant_entries.items():
            results.append({
                "id": entry_id,
                "content": self.knowledge_entries[entry_id],
                "relevance": data["score"],
                "matches": data["matches"]
            })
        
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:7]  # return top 7 aja
    
    def get_search_suggestions(self, failed_query):
        # kalo ga ketemu apa2, kasih saran keyword
        suggestions = []
        all_words = set()
        
        # kumpulin semua kata dari knowledge base
        for content in self.knowledge_entries.values():
            words = content.lower().replace(',', '').replace('.', '').split()
            all_words.update([word for word in words if len(word) > 3])
        
        # cari kata yang mirip sama query user
        query_words = failed_query.lower().split()
        for query_word in query_words:
            if len(query_word) > 2:
                matches = difflib.get_close_matches(query_word, all_words, n=3, cutoff=0.4)
                suggestions.extend(matches)
        
        return list(set(suggestions[:5]))  # max 5 saran unik

class DialogueGenerator:
    def __init__(self, model_path, n_gpu_layers):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.llm = self._load_model()
        self.messages = []  # Clean history - no panic responses
        self.complete_messages = []  # Full history - includes everything
        self.character_name = ""
        self.character_lie = ""
        self.character_weaknesses = []

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"--- ERROR ---")
            print(f"Model file not found at: {self.model_path}")
            print("Please update the MODEL_PATH variable in the script.")
            sys.exit(1)

        print("--- Loading Model... This may take a moment. ---")
        try:
            llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=32768,
                verbose=False
            )
            print("--- Model Loaded Successfully! ---")
            return llm
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def _generate_streamed_response(self):
        try:
            self.messages[-1]['content'] += " /no_think"  # biar ga ada thinking process
            output_stream = self.llm.create_chat_completion(
                messages=self.messages,
                max_tokens=32768,
                temperature=1.5,
                top_p=0.9,
                seed=-1,
                stream=True
            )

            full_response = ""
            for chunk in output_stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    token = delta['content']
                    full_response += token
                    print(token, end='', flush=True)
            
            print("\n")
            return full_response.strip()

        except Exception as e:
            print(f"\nError during inference: {e}")
            return "I... I can't think right now."
    
    def _llm_detect_weakness_hit(self, player_rebuttal):
        """
        Ask the LLM if the player's input targets any of the character's weaknesses.
        Return True if yes, else False.
        """
        if not self.character_weaknesses:
            return False

        weaknesses_list = "\n".join(f"- {w}" for w in self.character_weaknesses)
        detection_prompt = [
            {
                "role": "system",
                "content": (
                    "You are an analysis AI. Given a character's claim, their weaknesses, "
                    "and a player's rebuttal, determine if the rebuttal effectively targets any weakness "
                    "in relation to the specific claim being defended. "
                    "The weakness must be RELEVANT to challenging the claim. "
                    "Answer only with 'YES' or 'NO'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CHARACTER'S CLAIM: \"{self.character_lie}\"\n\n"
                    f"CHARACTER'S WEAKNESSES:\n{weaknesses_list}\n\n"
                    f"PLAYER'S REBUTTAL: \"{player_rebuttal}\"\n\n"
                    f"Does the rebuttal use relevant evidence/logic that effectively targets "
                    f"a weakness while challenging the specific claim \"{self.character_lie}\"? "
                    f"The evidence must actually contradict or undermine the claim, not just "
                    f"be mentioned in passing. /no_think"
                )
            }
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=detection_prompt,
                max_tokens=1000,
                temperature=0,
                top_p=1.0,
                seed=-1,
            )
            answer = response['choices'][0]['message']['content'].strip().upper()
            return "YES" in answer
        except Exception as e:
            print(f"LLM weakness detection error: {e}")
            # fallback to False on error
            return False


    def start_new_scenario(self, character, lie, weaknesses=None):
        self.character_name = character
        self.character_lie = lie
        self.character_weaknesses = weaknesses or []
        
        # bikin weakness descriptions buat prompt
        weakness_text = ""
        if self.character_weaknesses:
            weakness_text = f"\n\nCHARACTER WEAKNESSES (show these when pressured):\n"
            for weakness in self.character_weaknesses:
                weakness_text += f"- {weakness}\n"
        
        system_message = {
            "role": "system",
            "content": f"""You are roleplaying as {character} in a debate game. You are COMMITTED to defending this position: "{lie}"

PERSONALITY TRAITS:
- You genuinely believe your claim is true (or desperately need others to believe it)
- You're intelligent and will use logical-sounding arguments
- When challenged, you become more defensive but try to maintain credibility
- You may deflect, question the player's expertise, or provide alternative explanations
- You NEVER admit you're wrong, but you may show subtle signs of doubt when cornered

{weakness_text}

CRITICAL RULES:
- Respond ONLY with direct speech (no quotation marks, no actions, no narration)
- Keep responses conversational and natural (1-3 sentences max)
- Stay in character - you believe your position or need others to believe it
- When presented with strong counter-evidence, don't fold immediately - find ways to dismiss or reframe it
- Show personality: arrogance, nervousness, anger, or condescension as appropriate
- First person perspective only
- When the player hits your weaknesses, show the corresponding behavior but don't admit guilt

Remember: You are {character} defending "{lie}" - make it believable!"""
        }
        
        # Initialize both histories
        self.messages = [system_message]
        self.complete_messages = [system_message]

    def get_defensive_dialogue(self, player_rebuttal, skip_weakness_detection=False):
        weakness_hit = False
        if not skip_weakness_detection:
            weakness_hit = self._llm_detect_weakness_hit(player_rebuttal)

        # Add player input to COMPLETE history
        player_msg = {"role": "user", "content": f"Player: {player_rebuttal}"}
        self.complete_messages.append(player_msg)

        if weakness_hit:
            print("[DEBUG] Weakness hit detected!")
            semi_panic_messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are {self.character_name} and your weakness was just hit! You're in SEMI-PANIC mode. Respond with:"
                        "\n- Mild nervousness but still trying to sound confident"
                        "\n- Attempt to deflect with weak, illogical arguments"
                        "\n- Desperate but poorly thought-out explanations"
                        "\n- Slight defensive tone but still coherent speech"
                        "\n- Grasping at straws with unconvincing reasoning"
                        "\n- Maybe a bit of deflection like 'that's not relevant' or 'you don't understand'"
                        "\n- Still trying to maintain your position but the logic is obviously flawed"
                        "\n- 1-2 sentences maximum, don't ramble"
                    )
                },
                {
                    "role": "user",
                    "content": f"The player said: '{player_rebuttal}' /no_think"
                }
            ]
            try:
                output_stream = self.llm.create_chat_completion(
                    messages=semi_panic_messages, 
                    max_tokens=150,  
                    temperature=1.2,  
                    top_p=0.9,
                    seed=-1,
                    stream=True,
                )
                full_response = ""
                for chunk in output_stream:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        token = delta['content']
                        full_response += token
                        print(token, end='', flush=True)
                print()
                
                # Add semi-panic response to COMPLETE history only
                self.complete_messages.append({"role": "assistant", "content": full_response.strip()})
                
                return full_response.strip()
            except Exception as e:
                print(f"Semi-panic response error: {e}")
                # Changed fallback response to match semi-panic mode
                semi_panic_response = "That's... that's not how it works! You clearly don't understand the situation."
                self.complete_messages.append({"role": "assistant", "content": semi_panic_response})
                return semi_panic_response
        else:
            # Normal response logic remains the same
            base_prompt_content = (
                f"The player challenges you with: '{player_rebuttal}'\n\n"
                f"Remember, you are {self.character_name} defending that {self.character_lie}. Respond defensively but intelligently. /no_think"
            )
            
            # Add to BOTH histories for normal responses
            self.messages.append({"role": "user", "content": base_prompt_content})
            normal_response = self._generate_streamed_response()
            
            # Add response to BOTH histories
            self.messages.append({"role": "assistant", "content": normal_response})
            self.complete_messages.append({"role": "assistant", "content": normal_response})
            
            return normal_response


    def get_player_dialogue_choices(self, fact, claim):
        choice_generation_messages = [
            {
                "role": "system",
                "content": """You are a dialogue writer creating player response options for a debate game. The player wants to challenge a claim using a specific fact.

Create 4-5 dialogue options that:
1. Use the provided fact to challenge the claim
2. Vary in approach: direct confrontation, questioning, presenting evidence, logical reasoning
3. Range from aggressive to diplomatic in tone
4. Include one "trap" option that seems good but is actually weak or off-topic

Format your response as a JSON array of strings only. Each string should be a complete sentence the player could say.

Example response (inside `): `["Direct challenge using the fact", "Question that reveals the contradiction", "Diplomatic but firm rebuttal", "Aggressive confrontation", "Weak/off-topic option"]`"""
            },
            {
                "role": "user", 
                "content": f"CLAIM TO CHALLENGE: {claim}\nFACT TO USE: {fact}\n\nGenerate dialogue options that use this fact to challenge the claim."
            }
        ]
        
        try:
            response = self.llm.create_chat_completion(
                messages=choice_generation_messages,
                max_tokens=32768,
                temperature=1.,
                top_p=0.9,
                response_format={"type": "json_object"}
            )
            
            content = response['choices'][0]['message']['content']
            json_response = json.loads(content)
            
            # extract dialogue dari berbagai format JSON yang mungkin
            if isinstance(json_response, list):
                return json_response[:5]
            elif isinstance(json_response, dict):
                # cari array di dalam dict
                for value in json_response.values():
                    if isinstance(value, list) and len(value) > 0:
                        return value[:5]
            
            # fallback kalo parsing gagal
            return [
                f"But what about this: {fact}",
                f"How do you explain {fact.lower()}?",
                f"I have evidence that contradicts you: {fact.lower()}",
                "That's interesting, but I think you're mistaken.",
                "Let's talk about something else instead."  # opsi lemah sengaja
            ]
            
        except Exception as e:
            print(f"Error generating dialogue choices: {e}. Using default options.")
            return [
                f"But {fact.lower()}, doesn't that prove you wrong?",
                f"How do you explain {fact.lower()}?", 
                "I think you're not being truthful with me.",
                "That doesn't add up at all.",
                "I have my doubts about your story."
            ]

    def get_verdict(self):
        print("\n--- The debate is over. Analyzing the arguments... ---")
        
        judge_messages = [
            {
                "role": "system",
                "content": """You are an expert debate judge. Analyze the conversation between the Player and NPC where:
- The NPC was defending a false claim
- The Player was trying to expose the lie using facts

Evaluate based on:
1. STRENGTH OF ARGUMENTS: Did the player present compelling evidence?
2. LOGICAL CONSISTENCY: Did the player's points logically connect?
3. PERSUASIVE POWER: Would a reasonable person be convinced?
4. NPC RESISTANCE: How well did the NPC defend their position?

The player wins if they presented strong, logical arguments that would convince a reasonable person. The NPC wins if they successfully maintained doubt or deflected the player's attacks.

Respond with ONLY one word: "PLAYER" or "NPC" with the quotation marks"""
            },
            {
                "role": "user",
                "content": "Please judge this debate based on the strength of arguments and evidence presented."
            }
        ]
        
        # Use COMPLETE history for judgment - includes panic responses
        for msg in self.complete_messages[1:]:  # skip system message
            judge_messages.append(msg)

        try:
            response = self.llm.create_chat_completion(
                messages=judge_messages,
                max_tokens=32768,
                temperature=0.5
            )
            verdict = response['choices'][0]['message']['content'].strip().upper()
            
            if "PLAYER" in verdict:
                return "PLAYER"
            elif "NPC" in verdict:
                return "NPC"
            else:
                return "NPC"  # default ke NPC kalo ambiguous

        except Exception as e:
            print(f"Error getting verdict: {e}")
            return "NPC"


class Game:
    def __init__(self, dialogue_generator):
        self.dialogue_generator = dialogue_generator
        self.scenarios = self._load_scenarios()
    
    def _load_scenarios(self):
        # load scenarios dari file JSON external
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'scenarios.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                scenarios_data = json.load(f)
            return scenarios_data.get('scenarios', [])
        except FileNotFoundError:
            print("Scenarios file not found! Using basic fallback...")
            # fallback kalo file ga ada
            return [
                {
                    "character": "Generic NPC",
                    "lie": "I'm telling the truth",
                    "knowledge_type": "antique",
                    "rounds": 3,
                    "background": "Basic scenario",
                    "weaknesses": ["Gets nervous when questioned"]
                }
            ]
        except Exception as e:
            print(f"Error loading scenarios: {e}")
            return []

    def play(self):
        game_is_running = True
        while game_is_running:
            print("\n" + "="*60)
            print("RAG Interrogation Game: Choose a Scenario")
            print("="*60)
            for i, scenario in enumerate(self.scenarios):
                print(f"{i + 1}. Confront the {scenario['character']}")
                print(f"   Case: {scenario['background'][:60]}...")
                # tampilkan weakness hints kalo ada
                if 'weaknesses' in scenario:
                    print(f"   Hints: {len(scenario['weaknesses'])} known weaknesses")
            print("q. Quit Game")

            choice = input("\n> ")
            if choice.lower() == 'q':
                print("Thanks for playing!")
                break
            
            try:
                scenario_index = int(choice) - 1
                if 0 <= scenario_index < len(self.scenarios):
                    # tampilkan weakness info sebelum mulai
                    selected_scenario = self.scenarios[scenario_index]
                    if 'weaknesses' in selected_scenario:
                        print(f"\nKnown weaknesses of {selected_scenario['character']}:")
                        for weakness in selected_scenario['weaknesses']:
                            print(f"  - {weakness}")
                        input("\nPress Enter to begin the confrontation...")
                    
                    player_won = self.play_scenario(selected_scenario)
                    if player_won:
                        print("\nVICTORY! Your investigative skills are incredible!")
                        continue_choice = input("Play another scenario? [y/n] > ").lower()
                        if continue_choice != 'y':
                            game_is_running = False
                    else:
                        print("\nDEFEAT! The suspect's deception held firm.")
                        input("--- Press Enter to return to the menu ---")

                else:
                    print("Invalid choice, please try again.")
            except ValueError:
                print("Invalid input, please enter a number.")

    def play_scenario(self, scenario):
        # setup RAG components
        knowledge_base = KnowledgeBase(scenario['knowledge_type'])
        
        # pass weaknesses ke dialogue generator
        weaknesses = scenario.get('weaknesses', [])
        self.dialogue_generator.start_new_scenario(
            scenario['character'], 
            scenario['lie'],
            weaknesses
        )
        
        rounds = scenario['rounds']
        used_facts = set()  # track used facts/content here
        
        print("\n" + "-"*70)
        print(f"CONFRONTING: {scenario['character']}")
        print(f"THEIR CLAIM: \"{scenario['lie']}\"")
        print(f"BACKGROUND: {scenario['background']}")
        print(f"YOUR MISSION: Use research and evidence to expose their deception!")
        print(f"You have {rounds} rounds to build your case.")
        print("-"*70 + "\n")

        # opening dari NPC - no weakness detection needed
        print(f"{scenario['character']}: ", end='')
        opening = self.dialogue_generator.get_defensive_dialogue("Someone approaches you with questions about your recent claims.", 
        skip_weakness_detection=True)
        
        for i in range(rounds):
            print(f"\n{'='*25} ROUND {i + 1} of {rounds} {'='*25}")
            
            selected_fact = None
            search_attempts = 0
            max_searches = 3
            
            while selected_fact is None and search_attempts < max_searches:
                search_attempts += 1
                
                # fase RETRIEVAL
                print("RESEARCH PHASE:")
                if search_attempts > 1:
                    if search_attempts == 2:
                        print("WARNING: You're on your 2nd search attempt for this round!")
                    else:
                        print("FINAL WARNING: This is your last search attempt for this round!")
                
                print("Search your knowledge base for relevant information.")
                print("(Enter keywords related to what you want to investigate)")
                if search_attempts > 1:
                    print("TIP: Try different keywords, be more specific, or search for broader concepts")
                
                search_query = input("Search query > ").strip()
                if search_query.lower() == 'quit':
                    return False
                
                if not search_query:
                    print("You need to search for something!")
                    continue
                    
                # jalanin search
                search_results = knowledge_base.search(search_query)
                
                # Filter out already used facts
                filtered_results = [r for r in search_results if r['content'] not in used_facts]
                
                if not filtered_results:
                    print(f"No new relevant information found for '{search_query}' (all found facts have been used).")
                    
                    # Filter suggestions to exclude terms that would lead to already used facts
                    suggestions = knowledge_base.get_search_suggestions(search_query)

                    # Remove suggestions that are "close" to used facts entries
                    filtered_suggestions = []
                    for suggestion in suggestions:
                        # run search again to see if suggestion leads ONLY to used facts
                        suggestion_results = knowledge_base.search(suggestion)
                        # check if all suggestion_results are in used facts
                        if not suggestion_results:
                            # if nothing found, keep suggestion anyway
                            filtered_suggestions.append(suggestion)
                        else:
                            # If any new fact not used, keep suggestion
                            if any(result['content'] not in used_facts for result in suggestion_results):
                                filtered_suggestions.append(suggestion)

                    # kasih saran kalo ga ketemu
                    if filtered_suggestions:
                        print("Maybe try searching for:")
                        for suggestion in filtered_suggestions:
                            print(f"   - {suggestion}")
                    else:
                        print("No useful new search suggestions available.")

                    
                    if search_attempts < max_searches:
                        retry = input(f"\nTry a different search? ({max_searches - search_attempts} attempts left) [y/n] > ").lower()
                        if retry != 'y':
                            break
                    continue
                
                print(f"\nSEARCH RESULTS for '{search_query}' (Relevance ranked):")
                for idx, result in enumerate(filtered_results):
                    relevance_stars = "*" * min(int(result['relevance']), 5)
                    print(f"   {idx + 1}. {relevance_stars} {result['content']}")
                
                # player pilih fact atau search lagi
                print(f"\nOptions:")
                for idx in range(len(filtered_results)):
                    print(f"   {idx + 1}. Use fact {idx + 1}")
                
                if search_attempts < max_searches:
                    print(f"   s. Search again ({max_searches - search_attempts} attempts left)")
                print("   q. Quit game")
                
                choice = input(f"\nYour choice > ").strip().lower()
                
                if choice == 'quit' or choice == 'q':
                    return False
                elif choice == 's' and search_attempts < max_searches:
                    continue  # balik ke search lagi
                else:
                    try:
                        fact_idx = int(choice) - 1
                        if 0 <= fact_idx < len(filtered_results):
                            selected_fact = filtered_results[fact_idx]['content']
                            print(f"\nSelected: {selected_fact}")
                            used_facts.add(selected_fact)  # Add fact to used set here
                        else:
                            print("Invalid choice!")
                            if search_attempts < max_searches:
                                continue
                            else:
                                # fallback pick first unused fact
                                selected_fact = filtered_results[0]['content'] if filtered_results else None
                                if selected_fact:
                                    used_facts.add(selected_fact)
                    except ValueError:
                        print("Invalid input!")
                        if search_attempts < max_searches:
                            continue
                        else:
                            selected_fact = filtered_results[0]['content'] if filtered_results else None
                            if selected_fact:
                                used_facts.add(selected_fact)
            
            # kalo masih ga ada fact setelah 3x coba
            if selected_fact is None:
                print("No evidence selected! You fumble with your research...")
                selected_fact = "Some vague information that doesn't seem very useful"
                # Don't add vague fallback to used_facts to allow any other fact usage
            
            # fase GENERATION - bikin opsi dialogue berdasarkan fact
            print("\nGenerating dialogue options based on your research...")
            dialogue_options = self.dialogue_generator.get_player_dialogue_choices(selected_fact, scenario['lie'])
            
            print(f"\nHow do you want to use this evidence?")
            for idx, option in enumerate(dialogue_options):
                print(f"   {idx + 1}. {option}")
            
            dialogue_choice = input("\nChoose your approach > ")
            if dialogue_choice.lower() == 'quit':
                return False

            try:
                selected_dialogue = dialogue_options[int(dialogue_choice) - 1]
                print(f"\nYou: \"{selected_dialogue}\"")
                print(f"\n{scenario['character']}: ", end='')
                self.dialogue_generator.get_defensive_dialogue(selected_dialogue)
            except (ValueError, IndexError):
                print("\nInvalid choice! You stumble over your words...")
                print(f"\n{scenario['character']}: ", end='')
                self.dialogue_generator.get_defensive_dialogue("You seem to be struggling to make a point.")

        print("\n" + "="*70)
        verdict = self.dialogue_generator.get_verdict()
        print(f"FINAL VERDICT: {verdict} WINS!")
        
        if verdict == "PLAYER":
            print("VICTORY! Your research and strategic arguments were devastating!")
            print("You successfully used the RAG system to build an unbeatable case!")
        else:
            print("DEFEAT! They managed to deflect your evidence...")
            print("Try different search strategies and fact combinations next time.")
            
        print("="*70)
        return verdict == "PLAYER"

if __name__ == "__main__":
    dialogue_gen = DialogueGenerator(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS)
    game = Game(dialogue_generator=dialogue_gen)
    game.play()
	
