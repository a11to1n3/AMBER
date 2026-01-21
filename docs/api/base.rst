Base Classes
============

.. automodule:: ambr.base
   :members:
   :undoc-members:
   :show-inheritance:

The base module provides abstract base classes for creating custom models and agents.

BaseModel
---------

.. autoclass:: ambr.BaseModel
   :members:
   :undoc-members:

Abstract base class for all models. Inherit from this to create custom model types:

.. code-block:: python

   from ambr.base import BaseModel
   
   class CustomModel(BaseModel):
       def __init__(self, parameters):
           super().__init__(parameters)
           # Custom initialization
       
       def setup(self):
           # Model setup logic
           pass
       
       def step(self):
           # Step logic
           pass

BaseAgent
---------

.. autoclass:: ambr.BaseAgent
   :members:
   :undoc-members:

Abstract base class for all agents. Inherit from this to create custom agent types:

.. code-block:: python

   from ambr.base import BaseAgent
   
   class CustomAgent(BaseAgent):
       def __init__(self, model, agent_id):
           super().__init__(model, agent_id)
           # Custom initialization
           self.custom_property = "value"
       
       def step(self):
           # Agent behavior logic
           pass
       
       def custom_method(self):
           # Custom agent methods
           return self.custom_property

**Key Features:**

* Automatic parameter access via ``self.p``
* Model reference via ``self.model``
* Unique ID management
* Integration with data collection system 