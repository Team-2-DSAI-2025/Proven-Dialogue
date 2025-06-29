from dialogue_manager import DialogueManager

def main():
    model_path = "C:/Users/vince/.lmstudio/models/lmstudio-community/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q6_K.gguf"
    facts_path = "static/facts.json"
    manager = DialogueManager(model_path=model_path, facts_path=facts_path)

    npc_persona = "Grak, a proud and stubborn blacksmith who believes his work is the best in the world and is dismissive of foreign techniques."
    print(f"\n--- DIALOGUE START ---")
    print(f"You approach {npc_persona}.")

    # Phase 1: NPC makes a dubious claim
    claim, true_fact_id = manager.generate_npc_claim(npc_persona)
    true_fact = manager.fact_map[true_fact_id]
    print(f"\n{npc_persona.split(',')[0]} says: \"{claim}\"")

    # Phase 2: Player chooses to challenge or not
    print("\nWhat will you do?")
    print("[1] Let it slide.")
    print("[2] Challenge the claim.")
    choice = input("> ").strip()
    if choice != '2':
        print("\nYou decide not to press the issue. The conversation moves on.")
        print("--- DIALOGUE END ---")
        exit()

    # Phase 3: Retrieval Phase (mini-game simulation)
    print("\n--- RESEARCH PHASE ---")
    keywords = input("Enter keywords to search your lore journal> ").strip()
    results = manager.perform_research(keywords)
    if not results:
        print("Your research yields nothing useful.")
        was_successful = False
        player_rebuttal = "(You couldn't find any evidence.)"
    else:
        print("\nYour research found the following relevant facts:")
        for i, fact in enumerate(results, start=1):
            print(f"  [{i}] {fact['fact_text']}")
        print(f"  [{len(results)+1}] None of these.")
        try:
            selection = int(input("Select a fact to use as evidence> ").strip())
        except ValueError:
            selection = 0
        if selection < 1 or selection > len(results):
            selected_fact = None
        else:
            selected_fact = results[selection-1]
        # Determine success: player must select the exact fact that was contradicted
        if selected_fact and selected_fact['fact_id'] == true_fact_id:
            was_successful = True
            print("\nYou present the evidence from your archives...")
            player_rebuttal = manager.generate_rebuttal_option(claim, selected_fact)
            print(f"\nYOU: {player_rebuttal}")
        else:
            was_successful = False
            player_rebuttal = "(Your attempt to refute the claim falls flat.)"
            print("\nYou couldn't find the right evidence in time.")

    # Phase 4: Consequence - NPC reaction
    reaction = manager.generate_narrative_consequence(npc_persona, claim, player_rebuttal, was_successful, true_fact_id)
    if was_successful:
        print("\n(Your challenge was successful!)")
    else:
        print("\n(The NPC remains unconvinced.)")
    print(f"\n{npc_persona.split(',')[0]}: \"{reaction}\"")
    print("\n--- DIALOGUE END ---")



if __name__ == "__main__":
    main()