from fastapi import APIRouter, Depends, HTTPException
from main import mdb
from auth.helpers import get_current_user
from typing import Optional


activity_router = APIRouter()

@activity_router.get("/list-activity")
async def list_activities(
    space_id: Optional[str] = None,
    user=Depends(get_current_user),
):
    query = {}

    if space_id:
        # Specific space feed
        space = mdb.spaces.find_one({"_id": space_id})
        if not space:
            raise HTTPException(404, detail="Space not found")

        if user["_id"] != space["owner_id"] and user["_id"] not in space.get("members", []):
            raise HTTPException(403, detail="Not authorized to view activities")

        query["space_id"] = space_id
    else:
        user_spaces = mdb.spaces.find({
            "$or": [
                {"owner_id": user["_id"]},
                {"members": user["_id"]}
            ]
        }, {"_id": 1})

        allowed_space_ids = [s["_id"] for s in user_spaces]
        query["space_id"] = {"$in": allowed_space_ids}

    cursor = (
        mdb.activities.find(query)
        .sort("created_at", -1)
        .skip(0)
        .limit(10)
    )

    activities = []
    for act in cursor:
        activities.append({
            "id": str(act["_id"]),
            "space_id": act["space_id"],
            "actor": act["actor"],
            "type": act["type"],
            "description": act["description"],
            "created_at": act["created_at"],
        })

    return {"activities": activities}