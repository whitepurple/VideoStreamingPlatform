from django.contrib.auth.models import AbstractBaseUser,BaseUserManager,PermissionsMixin
from django.db import models 
class UserManager(BaseUserManager):    
    
    use_in_migrations = True    
    
    def create_user(self, username, password=None):        
   
        user = self.model(                      
            username = username
        )        
        user.set_password(password)        
        user.save(using=self._db)        
        return user     
    def create_superuser(self, username, password):        
       
        user = self.create_user(                      
            username = username,            
            password=password        
        )        
        user.is_admin = True        
        user.is_superuser = True        
        user.is_staff = True        
        user.save(using=self._db)        
        return user 

class DiceUser(AbstractBaseUser,PermissionsMixin):    
    
    objects = UserManager()
    email = models.EmailField(        
        max_length=255,        
    )   
    username = models.CharField(
        max_length=20,
        null=False,
        unique=True
    )
    is_active = models.BooleanField(default=True)    
    is_admin = models.BooleanField(default=False)    
    is_superuser = models.BooleanField(default=False)    
    is_staff = models.BooleanField(default=False)    
    date_joined = models.DateTimeField(auto_now_add=True)     
    USERNAME_FIELD = 'username'
    
    @property
    def profile(self):
        return  'img/s{}.png'.format(self.id%8+1)
        
    @property
    def pannel(self):
        return  'img/v{}.png'.format(self.id%7+1)