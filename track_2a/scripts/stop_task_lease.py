"""Stop this task container after work; cloud controller must still destroy it."""
import json
import os
import urllib.request

instance=os.environ.get('CONTAINER_ID','')
key=os.environ.get('CONTAINER_API_KEY','')
if not instance.isdigit() or not key:
    print('V15_SELF_STOP_UNAVAILABLE; cloud controller must stop/destroy task lease.',flush=True)
else:
    request=urllib.request.Request('https://console.vast.ai/api/v0/instances/'+instance+'/',
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},
        data=b'{"state":"stopped"}',method='PUT')
    try:
        with urllib.request.urlopen(request,timeout=30) as response:result=json.load(response)
        print('V15_TASK_SELF_STOP_REQUESTED',bool(result.get('success')),flush=True)
    except Exception as error:
        print('V15_SELF_STOP_FAILED',type(error).__name__,flush=True)
