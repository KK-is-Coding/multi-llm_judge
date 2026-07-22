
def get_user_feedback(judge_result: dict) -> dict:
    """
    Simulates or requests user feedback on the judge's selected answer.
    Returns a dict with status ('approved', 'rejected') and authorized_answer.
    """
    print("\n" + "="*50)
    print(f"JUDGE SELECTED ({judge_result.get('best_model')}):")
    print(f"RATIONALE: {judge_result.get('rationale')}")
    print("-" * 20)
    print(f"ANSWER: {judge_result.get('final_answer')}")
    print("="*50 + "\n")

    print("="*50 + "\n")

    # Direct feedback model
    print("Press [ENTER] to approve, or type your correction/feedback below:")
    user_input = input(">>> ").strip()

    if not user_input:
        # Empty input -> Approval
        return {
            "status": "approved",
            "final_answer": judge_result.get("final_answer")
        }
    else:
        # Non-empty input -> Correction (Explicit feedback)
        # We assume if they typed something, that IS the new answer.
        return {
            "status": "approved",
            "final_answer": user_input
        }
