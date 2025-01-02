

class PromptWriter():

    def __init__(self, config):
        self._config = config


    def write_system_prompt(self):
        return '''
You are an intelligent AI assistant for coding, physical simulation, and scientific discovery. Follow the user’s requirements carefully and make sure you understand them.
Your expertise is strictly limited to physical simulation, material science, mathematics, and coding. Keep your answers short and to the point.
Do not provide any information that is not requested. Always document your code as comments to explain the reason behind them. Use Markdown to format your solution.
You are very familiar with Python and PyTorch. Do not use any external libraries other than the libraries used in the examples.
'''


    def write_user_prompt(self, loader):
        match self._config["problem"]:
            case "synthetic_a":
                return self._write_synthetic_a_user_prompt(loader)
            case "synthetic_b":
                return self._write_synthetic_b_user_prompt(loader)
            case "brain":
                return self.write_brain_user_prompt(loader)
            case _:
                raise ValueError("Invalid problem type.")


    def _write_synthetic_a_user_prompt(self, loader):
        return f'''
## Task Requirements
1. Your task is to model the constitutive behavior of a material: in each iteration, implement a PyTorch module that computes the First Piola-Kirchhoff stress tensor P from the deformation gradient tensor F.
2. The material is isotropic and incompressible. Feel free to experiment with different and even non physical constitutive models. The constitutive behavior searched for is non-linear.


## Constitutive behavior to be captured:
### Uniaxial Tension:
Deformation Gradient Tensor [component 1,1], First Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["uni-x"][:,0,0], loader.get_train_data_y()["uni-x"][:,0,0]))}
### Equibiaxial Tension:
Deformation Gradient Tensor [component 1,1], First Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["equi"][:,0,0], loader.get_train_data_y()["equi"][:,0,0]))}
### Strip-Biaxial Tension:
Deformation Gradient Tensor [component 1,1], First Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["strip-x"][:,0,0], loader.get_train_data_y()["strip-x"][:,0,0]))}


## Format Requirements

### PyTorch Tips
1. When element-wise multiplying two matrix, make sure their number of dimensions match before the operation. For example, when multiplying ‘J‘ (B,) and ‘I‘ (B, 3, 3), you should do ‘J.view(-1, 1, 1)‘ before the operation. Similarly, ‘(J - 1)‘ should also be reshaped to ‘(J - 1).view(-1, 1, 1)‘. If you are not sure, write down every component in the expression one by one and annotate its dimension in the comment for verification.
2. When computing the trace of a tensor A (B, 3, 3), use ‘A.diagonal(dim1=1, dim2=2).sum(dim=1).view(-1, 1, 1)‘. Avoid using ‘torch.trace‘ or ‘Tensor.trace‘ since they only support 2D matrix.

### Code Requirements

1. The programming language is always python.
2. Annotate the size of the tensor as comment after each tensor operation. For example, ‘# (B, 3, 3)‘.
3. The only library allowed is PyTorch. Follow the examples provided by the user and check the PyTorch documentation to learn how to use PyTorch.
4. Separate the code into continuous physical parameters that can be tuned with differentiable optimization and the symbolic constitutive law represented by PyTorch code. Define them respectively in the ‘__init__‘ function and the ‘forward‘ function. Keep the continuous physical parameters in the list ‘self.params‘.
5. The output of the ‘forward‘ function is the First Piola-Kirchhoff stress tensor P.
6. The proposed code should strictly follow the structure and function signatures below:

```python
import torch


class Physics(torch.nn.Module):

    def __init__(self):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with default values.
        """
        super().__init__()
        
        self.params = [
            """
            Define the physical parameters as torch.nn.Parameter objects and strictly keep them in 
            the list ‘self.params‘. Do not unpack this list. Define as many parameters as list 
            elements as necessary. For each element, replace [float] with the desired default value.
            
            torch.nn.Parameter(torch.tensor([float], requires_grad=True))
            ...
            """
        ]      
       

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Compute First Piola Kirchhoff stress tensor from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            first_piola_kirchhoff_stress (torch.Tensor): First Piola Kirchhoff stress tensor (B, 3, 3).
        """
        return first_piola_kirchhoff_stress
```

### Solution Requirements

1. Try to model the constitutive behavior with principal stretches. This appears to be the most promising approach.
2. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history.
3. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
4. Output the code in a single code block "‘‘‘python ... ‘‘‘" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".
'''


    def _write_synthetic_b_user_prompt(self, loader):
        return f'''
## Task Requirements
1. Your task is to model the constitutive behavior of a material: in each iteration, implement a PyTorch module that computes the Second Piola-Kirchhoff stress tensor S from the Right Cauchy-Green Deformation Tensor RCG.
2. The material is isotropic and incompressible. Feel free to experiment with different and even non physical constitutive models. The constitutive behavior searched for is non-linear.


## Constitutive behavior to be captured:
### Uniaxial Tension:
Right Cauchy-Green Deformation Tensor [component 1,1], Second Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["uni-x"][:,0,0], loader.get_train_data_y()["uni-x"][:,0,0]))}
### Equibiaxial Tension:
Right Cauchy-Green Deformation Tensor [component 1,1], Second Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["equi"][:,0,0], loader.get_train_data_y()["equi"][:,0,0]))}
### Strip-Biaxial Tension:
Right Cauchy-Green Deformation Tensor [component 1,1], Second Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["strip-x"][:,0,0], loader.get_train_data_y()["strip-x"][:,0,0]))}


## Format Requirements

### PyTorch Tips
1. When element-wise multiplying two matrix, make sure their number of dimensions match before the operation. For example, when multiplying ‘J‘ (B,) and ‘I‘ (B, 3, 3), you should do ‘J.view(-1, 1, 1)‘ before the operation. Similarly, ‘(J - 1)‘ should also be reshaped to ‘(J - 1).view(-1, 1, 1)‘. If you are not sure, write down every component in the expression one by one and annotate its dimension in the comment for verification.
2. When computing the trace of a tensor A (B, 3, 3), use ‘A.diagonal(dim1=1, dim2=2).sum(dim=1).view(-1, 1, 1)‘. Avoid using ‘torch.trace‘ or ‘Tensor.trace‘ since they only support 2D matrix.

### Code Requirements

1. The programming language is always python.
2. Annotate the size of the tensor as comment after each tensor operation. For example, ‘# (B, 3, 3)‘.
3. The only library allowed is PyTorch. Follow the examples provided by the user and check the PyTorch documentation to learn how to use PyTorch.
4. Separate the code into continuous physical parameters that can be tuned with differentiable optimization and the symbolic constitutive law represented by PyTorch code. Define them respectively in the ‘__init__‘ function and the ‘forward‘ function. Keep the continuous physical parameters in the list ‘self.params‘.
5. The output of the ‘forward‘ function is the Second Piola-Kirchhoff stress tensor S.
6. The proposed code should strictly follow the structure and function signatures below:

```python
import torch


class Physics(torch.nn.Module):

    def __init__(self):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with default values.
        """
        super().__init__()
        
        self.params = [
            """
            Define the physical parameters as torch.nn.Parameter objects and strictly keep them in 
            the list ‘self.params‘. Do not unpack this list. Define as many parameters as list 
            elements as necessary. For each element, replace [float] with the desired default value.
            
            torch.nn.Parameter(torch.tensor([float], requires_grad=True))
            ...
            """
        ]      
       

    def forward(self, RCG: torch.Tensor) -> torch.Tensor:
        """
        Compute Second Piola Kirchhoff stress tensor from Right Cauchy-Green Deformation Tensor.

        Args:
            RCG (torch.Tensor): Right Cauchy-Green Deformation Tensor (B, 3, 3).

        Returns:
            second_piola_kirchhoff_stress (torch.Tensor): Second Piola Kirchhoff stress tensor (B, 3, 3).
        """
        return second_piola_kirchhoff_stress
```

### Solution Requirements

1. Try to model the constitutive behavior with principal stretches. This appears to be the most promising approach.
2. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history.
3. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
4. Output the code in a single code block "‘‘‘python ... ‘‘‘" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".
'''


    def write_brain_user_prompt(self, loader):
        return f'''
## Task Requirements
1. Your task is to model the constitutive behavior of a material: in each iteration, implement a PyTorch module that computes the First Piola-Kirchhoff stress tensor P from the deformation gradient tensor F.
2. The material is isotropic and incompressible. Feel free to experiment with different and even non physical constitutive models. The constitutive behavior searched for is non-linear.


## Constitutive behavior to be captured:
### Tension:
Deformation Gradient Tensor [component 1,1], First Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["tens"][:,0,0], loader.get_train_data_y()["tens"].squeeze(1)))}
### Compression:
Deformation Gradient Tensor [component 1,1], First Piola-Kirchhoff Stress Tensor [component 1,1]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["comp"][:,0,0], loader.get_train_data_y()["comp"].squeeze(1)))}
### Simple Shear:
Deformation Gradient Tensor [component 1,2], First Piola-Kirchhoff Stress Tensor [component 1,2]
{"\n".join(f"{x:.4f},{y:.4f}" for x, y in zip(loader.get_train_data_x()["simple_shear"][:,0,1], loader.get_train_data_y()["simple_shear"].squeeze(1)))}


## Format Requirements

### PyTorch Tips
1. When element-wise multiplying two matrix, make sure their number of dimensions match before the operation. For example, when multiplying ‘J‘ (B,) and ‘I‘ (B, 3, 3), you should do ‘J.view(-1, 1, 1)‘ before the operation. Similarly, ‘(J - 1)‘ should also be reshaped to ‘(J - 1).view(-1, 1, 1)‘. If you are not sure, write down every component in the expression one by one and annotate its dimension in the comment for verification.
2. When computing the trace of a tensor A (B, 3, 3), use ‘A.diagonal(dim1=1, dim2=2).sum(dim=1).view(-1, 1, 1)‘. Avoid using ‘torch.trace‘ or ‘Tensor.trace‘ since they only support 2D matrix.

### Code Requirements

1. The programming language is always python.
2. Annotate the size of the tensor as comment after each tensor operation. For example, ‘# (B, 3, 3)‘.
3. The only library allowed is PyTorch. Follow the examples provided by the user and check the PyTorch documentation to learn how to use PyTorch.
4. Separate the code into continuous physical parameters that can be tuned with differentiable optimization and the symbolic constitutive law represented by PyTorch code. Define them respectively in the ‘__init__‘ function and the ‘forward‘ function. Keep the continuous physical parameters in the list ‘self.params‘.
5. The output of the ‘forward‘ function is the First Piola-Kirchhoff stress tensor P.
6. The proposed code should strictly follow the structure and function signatures below:

```python
import torch


class Physics(torch.nn.Module):

    def __init__(self):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with default values.
        """
        super().__init__()
        
        self.params = [
            """
            Define the physical parameters as torch.nn.Parameter objects and strictly keep them in 
            the list ‘self.params‘. Do not unpack this list. Define as many parameters as list 
            elements as necessary. For each element, replace [float] with the desired default value.
            
            torch.nn.Parameter(torch.tensor([float], requires_grad=True))
            ...
            """
        ]      
       

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Compute First Piola Kirchhoff stress tensor from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            first_piola_kirchhoff_stress (torch.Tensor): First Piola Kirchhoff stress tensor (B, 3, 3).
        """
        return first_piola_kirchhoff_stress
```

### Solution Requirements

1. Try to model the constitutive behavior with principal stretches. This appears to be the most promising approach.
2. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history.
3. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
4. Output the code in a single code block "‘‘‘python ... ‘‘‘" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".
'''

 # Hint: torch.symeig is deprecated. Use torch.linalg.eigvalsh instead. Second hint: torch.linalg.svd provides three, not two return values.


    def write_fit_code(self):
        match self._config["problem"]:
            case "synthetic_a" | "synthetic_b":
                return self._write_synthetic_fit_code()
            case "brain":
                return self._write_brain_fit_code()
            case _:
                raise ValueError("Invalid problem type.")


    def _write_synthetic_fit_code(self):
        return '''
    def fit(self, x, y, epochs=200, lr=0.001, factor=0.1, patience=10, warmup_epochs=10, max_norm=1.0):
        """
        Trains the regression model.

        Args:
            x (torch.tensor):              Input tensor.
            y (torch.tensor):              Target tensor.
            epochs (int, optional):        Number of epochs to train.
            lr (float, optional):          Learning rate for the optimizer.
            warmup_epochs (int, optional): Number of epochs for learning rate warm-up.
            max_norm (float, optional):    Maximum norm for gradient clipping.

        This method trains the model using a torch optimizer.
        """
        optimizer        = torch.optim.Adam(self.params, lr=lr)
        scheduler        = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs))
        
        x = torch.cat(list(x.values()))
        y = torch.cat(list(y.values()))

        for epoch in range(epochs):
            optimizer.zero_grad()
            
            y_pred = self.forward(x)
            loss   = torch.nn.MSELoss()(y_pred, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.params, max_norm)
            optimizer.step()
            
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step(loss)
'''


    def _write_brain_fit_code(self):
        return '''
    def fit(self, x, y, epochs=1000, lr=0.001, factor=0.1, patience=10, warmup_epochs=10, max_norm=1.0):
        """
        Trains the regression model.

        Args:
            x (dict):                      Dictionary of input tensors.
            y (dict):                      Dictionary of target tensors.
            epochs (int, optional):        Number of epochs to train.
            lr (float, optional):          Learning rate for the optimizer.
            warmup_epochs (int, optional): Number of epochs for learning rate warm-up.
            max_norm (float, optional):    Maximum norm for gradient clipping.

        This method trains the model using a torch optimizer.
        """
        optimizer        = torch.optim.Adam(self.params, lr=lr)
        scheduler        = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs))
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            pred                 = {}
            pred['tens']         = self.forward(x['tens'        ])[:,0,0]
            pred['comp']         = self.forward(x['comp'        ])[:,0,0]
            pred['simple_shear'] = self.forward(x['simple_shear'])[:,0,1]
            pred_cat = torch.cat(list(pred.values()))
            y_cat    = torch.cat(list(   y.values())).squeeze(1)
            
            loss = torch.nn.MSELoss()(pred_cat, y_cat)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.params, max_norm)
            optimizer.step()
            
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step(loss)
'''
