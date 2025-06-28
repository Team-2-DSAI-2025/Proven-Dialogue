from dialogue_manager import DialogueManager

def main():
    # --- SETUP ---
    model_path = "C:/Users/vince/.lmstudio/models/lmstudio-community/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q6_K.gguf"
    facts_path = "static/facts.json"
    manager = DialogueManager(model_path=model_path, facts_path=facts_path)

    # --- GAME LOOP ---
    npc_persona = "Grak, a proud and stubborn blacksmith who believes his work is the best in the world and is dismissive of foreign techniques."
    
    print(f"\n--- DIALOGUE START ---")
    print(f"You approach {npc_persona}.")

    # FASE 1: THE DUBIOUS CLAIM
    claim, true_fact_id = manager.generate_npc_claim(npc_persona)
    true_fact_obj = manager.fact_map[true_fact_id]
    print(f"\n{npc_persona.split(',')[0]} scoffs and says: '{claim}'")

    # FASE 2: THE CHALLENGE
    print("\n[1] Let the claim slide.")
    print("[2] 'I'm not so sure about that...' (Challenge the claim)")
    choice = input("> ")

    if choice == '1':
        print("\nYou decide not to press the issue. The conversation moves on.")
        print("\n--- DIALOGUE END ---")
        return

    # FASE 3: THE RETRIEVAL PHASE (MINI-GAME)
    print("\n--- RESEARCH PHASE ---")
    print("Search your archives for information.")
    search_keywords = input("Keywords> ")
    
    search_results = manager.perform_research(search_keywords)

    if not search_results:
        print("Your research yields nothing of use.")
        # Di sini, kita bisa langsung memanggil konsekuensi kegagalan
        was_successful = False
        player_rebuttal_text = "(You found no evidence to back up your suspicion.)"
    else:
        print("\nYour research turns up the following entries:")
        for i, fact in enumerate(search_results):
            # Dalam game sungguhan, Anda mungkin hanya menampilkan sebagian kecil teks.
            print(f"  [{i+1}] {fact['fact_text']}")
        
        print(f"  [{len(search_results) + 1}] None of these seem relevant.")
        
        try:
            selection = int(input("Select evidence to use> ")) - 1
            if selection == len(search_results): # Pilihan "None"
                 selected_fact = None
            else:
                 selected_fact = search_results[selection]
        except (ValueError, IndexError):
            selected_fact = None

        # Evaluasi sekarang sederhana: apakah pemain menemukan fakta yang BENAR?
        was_successful = (selected_fact is not None and selected_fact['fact_id'] == true_fact_id)

        if was_successful:
            # FASE 4: THE GENERATION PHASE
            print("\nGenerating a rebuttal based on your findings...")
            player_rebuttal_text = manager.generate_rebuttal_option(claim, selected_fact)
            print(f"\nNew dialogue option available: ")
            print(f"YOU: {player_rebuttal_text}")
        else:
            player_rebuttal_text = "(You failed to find the correct piece of evidence.)"


    # FASE 5: THE CONSEQUENCE
    # Panggil fungsi konsekuensi dengan hasil dari fase riset
    consequence = manager.generate_narrative_consequence(
        npc_persona, claim, player_rebuttal_text, was_successful, true_fact_id
    )
    
    if was_successful:
        print(f"\nYou present your evidence convincingly!")
    else:
        print(f"\nYou couldn't find the right information, and your challenge falls flat.")
        
    print(f"\n{npc_persona.split(',')[0]} responds: '{consequence}'")
    print("\n--- DIALOGUE END ---")


if __name__ == "__main__":
    main()