from diceuser.models import DiceUser
    
def get_users_orderby_streaming():
    users = DiceUser.objects\
                    .filter(is_superuser=False)\
                    .order_by('-stream__started_at').all()
    return users