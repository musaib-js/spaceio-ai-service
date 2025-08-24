from main import mdb    
from auth.helpers import now_utc

def update_space_last_activity(space_id: str):
    update_data = {"last_activity": now_utc()}
    mdb.spaces.update_one({"_id": space_id}, {"$set": update_data})
    
def _log_activity(space_id: str, actor: dict, type_: str, description: str):
    mdb.activities.insert_one(
        {
            "space_id": space_id,
            "actor": {
                "id": actor["_id"],
                "name": actor.get("name"),
                "team": actor.get("team"),
            },
            "type": type_,
            "description": description,
            "created_at": now_utc(),
        }
    )