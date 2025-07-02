import inspect

'''
# 这段代码的目的是在创建一个配置对象时，自动实例化它的所有嵌套类。
# 这样你就不需要手动实例化每个嵌套类，特别适用于配置结构较为复杂、
# 层级较深的情况。通过这种方式，你的代码变得更加简洁，管理起来也更方便。
class SomeConfig(BaseConfig):
    class SubConfig:
        def __init__(self):
            self.value = 42

config = SomeConfig()
print(config.SubConfig.value)  # 输出 42
'''
class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod # 通过递归的方式初始化类中所有成员变量的类型为类实例，简化类的构造过程
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)