from django.shortcuts import render
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest
from DocChatRAG.chat.chain import build_chain

chain = build_chain()

@csrf_exempt
def chat_view(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST only")

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    session_id = body.get("session_id")
    user_input = body.get("message")
    if not session_id or not user_input:
        return HttpResponseBadRequest("session_id and message are required")

    try:
        answer = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        return JsonResponse({"session_id": session_id, "answer": answer})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
