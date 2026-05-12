# -*- coding: utf-8 -*-
"""
 avi  task prompt（）
 eval_task1_only / eval_full_trajectory_batch
"""
# task_id -> prompt ( taskN_xxx.avi )
TASK_PROMPTS = {
    "task1": "pick and place cookies in basket and place tomato sauce into where cookies was placed",
    "task2": "pick and place butter in basket and place popcorn into where butter was placed",
    "task3": "pick and place cream in basket and place chocolate into where cream was placed",
    "task4": "place butter into drawer where have object",
    "task5": "place butter into drawer where have not object",
    "task6": "pour tomato sauce twice on cookies and place tomato sauce into bowl drainer",
    "task7": "pour tomato sauce twice on frypan and place tomato sauce in bowl drainer",
    "task8": "pick chocolate in frypan pour tomato sauce twice and place tomato sauce into bowl drainer",
    "task9": "pick butter in frypan pour tomato sauce twice",
    "task10": "pour wine into white yellow mug twice",
    "task11": "place butter into top drawer and place butter into another closest drawer",
    "task12": "place cookies into middle drawer and place chocolate into drawer where cookies have placed",
    "task13": "place butter into middle drawer and place cookies into drawer where butter have placed",
    "task14": "place cookies into top drawer and place butter into another closest drawer",
    "task15": "place butter into frypan and pour milk twice into frypan",
    "task16": "pour milk in red coffee mug twice and place milk in bowl drainer test",
    "task17": "place butter into middle drawer and place chocolate into drawer where butter have placed",
    "task18": "pick chocolate and butter from cabinet1 to cabinet2",
    "task19": "pick tomato sauce and milk and orange juice from cabinet1 to cabinet2",
    "task20": "pick and place cookies into microwave and place chocolate into where cookies was placed",
    "task21": "pick and place butter into microwave and place chocolate into where butter was placed",
    "task22": "pour tomato sauce over cookies twice and place cookies into microwave",
    "task23": "pick and place cream into microwave and place popcorn into where cream was placed",
    "task24": "pick and place cookies into microwave and place popcorn into where cookies was placed",
    "task25": "pick and place butter cream from plate1 to plate2",
    "task26": "pick and place chocolate pudding and cream cheese",
}


def get_prompt(task_id: str, fallback_task_name: str = "") -> str:
    """ task  prompt， fallback_task_name.replace('_', ' ')"""
    if task_id in TASK_PROMPTS:
        return TASK_PROMPTS[task_id]
    return fallback_task_name.replace("_", " ") if fallback_task_name else ""
