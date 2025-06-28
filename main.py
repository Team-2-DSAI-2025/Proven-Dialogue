from dialogue_manager import DialogueManager

def main():
    # --- SETUP ---
    # Ensure you have downloaded the model and placed it in the 'models' directory
    model_path = "C:/Users/vince/.lmstudio/models/lmstudio-community/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q6_K.gguf"

    facts_path = "static/facts.json"
    
    manager = DialogueManager(model_path=model_path, facts_path=facts_path)

    # --- GAME LOOP EXAMPLE ---
    npc_persona = "Grak, a proud and stubborn blacksmith who believes his work is the best in the world and is dismissive of foreign techniques."
    
    print(f"\n--- DIALOGUE START ---")
    print(f"You approach {npc_persona}.")

    # 1. Generate NPC Claim

    claim, fact_id = manager.generate_npc_claim(npc_persona)
    print(f"\n{npc_persona.split(',')[0]} scoffs and says: '{claim}'")



    # 2. Get Player Rebuttal
    print("\nFormulate your rebuttal based on your research.")
    player_rebuttal = input("> ")

    # 3. Evaluate Rebuttal
    was_successful, score = manager.evaluate_rebuttal(player_rebuttal, fact_id)
    print(f"\n")

    # 4. Generate Consequence
    consequence = manager.generate_narrative_consequence(
        npc_persona, claim, player_rebuttal, was_successful, fact_id
    )
    
    if was_successful:
        print(f"\nYour argument is convincing!")
    else:
        print(f"\nYour argument fails to convince him.")
        
    print(f"\n{npc_persona.split(',')[0]} responds: '{consequence}'")
    print("\n--- DIALOGUE END ---")

if __name__ == "__main__":
    main()