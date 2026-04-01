"""AI Chat endpoint — edit feature tree, describe parts, suggest dimensions.

Uses Claude CLI via ai.py with all existing injection/budget protections.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from mesh2param.ai import call_ai, is_available, get_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
async def chat(body: dict):
    """Send a message to Claude with feature tree context.

    Request: {"message": "make hole 1 larger", "context": {...}}
    Response: {"reply": "...", "edit": null | {...}}
    """
    if not is_available():
        return {"reply": "AI is unavailable — Claude CLI not installed or not authenticated.", "edit": None}

    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message required")

    context = body.get("context", {})
    steps = context.get("steps", [])
    mesh_volume = context.get("mesh_volume", 0)
    cad_volume = context.get("cad_volume", 0)

    # Build context summary for the prompt
    steps_text = ""
    for s in steps:
        steps_text += f"  Step {s.get('index', '?')}: {s.get('name', '?')}"
        if s.get("volume"):
            steps_text += f" ({s['volume']} mm³)"
        steps_text += "\n"

    prompt = f"""You are a CAD assistant for mesh2param. The user is viewing a reconstructed
parametric model. Answer questions about it or suggest edits.

## Current Model
Mesh volume: {mesh_volume} mm³
CAD volume: {cad_volume} mm³
Construction steps:
{steps_text}

## Rules
- If the user asks to modify a feature (resize hole, change depth, etc.),
  include an "edit" object in your response with the modification.
- Edit format: {{"step": <index>, "action": "resize_hole"|"change_depth"|"remove", "value": <number>}}
- If it's just a question, set "edit" to null.
- Keep responses concise (2-3 sentences max).
- Respond with ONLY valid JSON: {{"reply": "...", "edit": null | {{...}}}}

## User Message
{message}"""

    try:
        response = call_ai(prompt, timeout=30)

        # Parse JSON response
        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])

        try:
            data = json.loads(text)
            return {
                "reply": data.get("reply", text),
                "edit": data.get("edit"),
                "usage": get_usage(),
            }
        except json.JSONDecodeError:
            return {"reply": text, "edit": None, "usage": get_usage()}

    except RuntimeError as e:
        return {"reply": f"AI error: {e}", "edit": None, "usage": get_usage()}


@router.post("/analyze-mesh")
async def analyze_mesh(body: dict):
    """Analyze mesh dimensions and suggest clean engineering values.

    Request: {"bbox_min": [...], "bbox_max": [...], "volume": ..., "num_faces": ..., "features": [...]}
    Response: {"part_type": "bracket", "suggested_dimensions": {...}, "scale_factor": 1.0, "notes": "..."}
    """
    if not is_available():
        return {
            "part_type": "unknown",
            "suggested_dimensions": {},
            "scale_factor": 1.0,
            "notes": "AI unavailable — skipping dimension analysis",
            "skip": True,
        }

    bbox_min = body.get("bbox_min", [0, 0, 0])
    bbox_max = body.get("bbox_max", [0, 0, 0])
    volume = body.get("volume", 0)
    num_faces = body.get("num_faces", 0)
    features = body.get("features", [])

    dims = [round(bbox_max[i] - bbox_min[i], 2) for i in range(3)]

    features_text = ""
    for f in features[:10]:
        features_text += f"  {f.get('type', '?')}: depth={f.get('depth', 0)}"
        if f.get("sketch", {}).get("radius"):
            features_text += f", radius={f['sketch']['radius']}"
        features_text += "\n"

    prompt = f"""Analyze this mechanical part and suggest clean engineering dimensions.

## Mesh Stats
Bounding box: {dims[0]} x {dims[1]} x {dims[2]} mm
Volume: {round(volume, 1)} mm³
Faces: {num_faces}

## Recognized Features
{features_text}

## Task
1. Identify the part type (bracket, plate, mount, housing, etc.)
2. Check if dimensions look like real engineering values or arbitrary scan data
3. If dimensions are already clean (round numbers, standard sizes), suggest scale_factor=1.0
4. If dimensions look arbitrary, suggest rounding to nearest standard sizes
5. Note any holes that should be standard drill sizes (M3=3.4mm, M4=4.5mm, M5=5.5mm, M6=6.6mm, etc.)

Respond with ONLY valid JSON:
{{
  "part_type": "bracket",
  "suggested_dimensions": {{
    "width": 160.0,
    "height": 140.0,
    "thickness": 12.0,
    "hole_diameters": [5.0, 5.0, 5.0, 5.0]
  }},
  "scale_factor": 1.0,
  "notes": "Dimensions look like real values. Holes at 5mm match M5 clearance."
}}"""

    try:
        response = call_ai(prompt, timeout=30)

        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])

        try:
            data = json.loads(text)
            data["skip"] = False
            data["usage"] = get_usage()
            return data
        except json.JSONDecodeError:
            return {
                "part_type": "unknown",
                "suggested_dimensions": {},
                "scale_factor": 1.0,
                "notes": text[:200],
                "skip": True,
                "usage": get_usage(),
            }

    except RuntimeError as e:
        return {
            "part_type": "unknown",
            "suggested_dimensions": {},
            "scale_factor": 1.0,
            "notes": f"AI error: {e}",
            "skip": True,
            "usage": get_usage(),
        }
