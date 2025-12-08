from backend.services.conversation_service import get_conversation_service

svc = get_conversation_service('default')
convs = svc.get_user_conversations()
print(f'Total conversations: {len(convs)}')
for c in convs:
    conv_id = c.get('conversation_id', 'N/A')
    title = c.get('title', 'N/A')
    msg_count = len(c.get('messages', []))
    print(f"  - {conv_id}: {title} ({msg_count} messages)")
    
# Test fetching one
if convs:
    first_conv = convs[0]
    conv_id = first_conv['conversation_id']
    print(f"\nTesting get_conversation_messages for: {conv_id}")
    messages = svc.get_conversation_messages(conv_id)
    print(f"Got {len(messages)} messages: {messages}")
