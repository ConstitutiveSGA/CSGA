

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


    def write_user_prompt(self):
        match self._config["problem"]:
            case "treloar_uniaxial_tension" | \
                 "treloar_biaxial_tension"  | \
                 "treloar_shear":
                return self._write_treloar_user_prompt()
            case "synthetic_a_uniaxial_tension" | \
                 "synthetic_a_biaxial_tension"  | \
                 "synthetic_a_shear":
                return self._write_synthetic_a_and_brain_user_prompt()
            case "synthetic_b_uniaxial_tension" | \
                 "synthetic_b_biaxial_tension"  | \
                 "synthetic_b_shear":
                return self._write_synthetic_b_user_prompt()
            case "brain_tension"     | \
                 "brain_compression" | \
                 "brain_shear":
                return self._write_synthetic_a_and_brain_user_prompt()
            case _:
                raise ValueError("Invalid problem type.")


    def _write_treloar_user_prompt(self):
        return '''
### Code Requirements

1. The programming language is always python. 
2. The only library allowed is PyTorch. Follow the examples provided by the user and check the PyTorch documentation to learn how to use PyTorch.
3. Separate the code into continuous physical parameters that can be tuned with differentiable optimization and the symbolic constitutive law represented by PyTorch code. Define them respectively in the ‘__init__‘ function and the ‘forward‘ function. Keep the continuous physical parameters in the list ‘self.params‘.
4. The input and output of the ‘forward‘ function (cauchy_green_deformation and kirchhoff_stress) are batches of scalar values, stored in torch tensors to allow parameter optimization. Do not try to perform matrix or tensor operations like ‘trace‘ on them. Treat them as what they are - stacked scalars!
5. The proposed code should strictly follow the structure and function signatures below:

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
       

    def forward(self, cauchy_green_deformation) -> torch.tensor:
        """
        Compute Second Piola-Kirchhoff Stress Values from Right Cauchy-Green Deformation Values. 

        Args:
            x (torch.tensor): Right Cauchy-Green Deformation Values. Scalars stored in 1D torch tensor.

        Returns:
            torch.tensor: Second Piola-Kirchhoff Stress Values. Scalars stored in 1D torch tensor.
        """
        return second_piola_kirchhoff_stress
```

### Solution Requirements

1. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history. Be creative! Do not try any approach you find in a previous iteration a second time.
2. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
3. Output the code in a single code block "‘‘‘python ... ‘‘‘" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".
'''


    def _write_synthetic_a_and_brain_user_prompt(self):
        return '''
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

1. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history.
2. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
3. Output the code in a single code block "‘‘‘python ... ‘‘‘" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".
'''


    def _write_synthetic_b_user_prompt(self):
        return '''
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

1. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history.
2. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
3. Output the code in a single code block "‘‘‘python ... ‘‘‘" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".
'''


    def write_fit_code(self):
        match self._config["problem"]:
            case "treloar_uniaxial_tension" | \
                 "treloar_biaxial_tension"  | \
                 "treloar_shear":
                return self._write_treloar_fit_code()
            case "synthetic_a_uniaxial_tension" | \
                 "synthetic_a_biaxial_tension"  | \
                 "synthetic_a_shear"            | \
                 "synthetic_b_uniaxial_tension" | \
                 "synthetic_b_biaxial_tension"  | \
                 "synthetic_b_shear":
                return self._write_synthetic_fit_code()
            case "brain_tension":
                return self._write_brain_tension_fit_code()
            case "brain_compression":
                return self._write_brain_compression_fit_code()
            case "brain_shear":
                return self._write_brain_shear_fit_code()
            case _:
                raise ValueError("Invalid problem type.")


    def _write_treloar_fit_code(self):
        return '''
    def fit(self, x, y, epochs=200, lr=0.001, factor=0.1, patience=10):
        """
        Trains the regression model.

        Args:
            x (torch.tensor):       Input tensor.
            y (torch.tensor):       Target tensor.
            epochs (int, optional): Number of epochs to train.
            lr (float, optional):   Learning rate for the optimizer.

        This method trains the model using a torch optimizer.
        """
        optimizer = torch.optim.Adam(self.params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss   = torch.nn.MSELoss()(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
'''


    def _write_synthetic_fit_code(self):
        return '''
    def fit(self, x, y, epochs=200, lr=0.001, factor=0.1, patience=10):
        """
        Trains the regression model.

        Args:
            x (torch.tensor):       Input tensor.
            y (torch.tensor):       Target tensor.
            epochs (int, optional): Number of epochs to train.
            lr (float, optional):   Learning rate for the optimizer.

        This method trains the model using a torch optimizer.
        """
        optimizer = torch.optim.Adam(self.params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss   = torch.nn.MSELoss()(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
'''

    def _write_brain_tension_fit_code(self):
        return '''
    def fit(self, x, y, epochs=200, lr=0.001, factor=0.1, patience=10):
        """
        Trains the regression model.

        Args:
            x (torch.tensor):       Input tensor.
            y (torch.tensor):       Target tensor.
            epochs (int, optional): Number of epochs to train.
            lr (float, optional):   Learning rate for the optimizer.

        This method trains the model using a torch optimizer.
        """
        optimizer = torch.optim.Adam(self.params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)[:,0,0].unsqueeze(1) # extract sigma_1_1
            loss   = torch.nn.MSELoss()(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
'''


    def _write_brain_compression_fit_code(self):
        return '''
    def fit(self, x, y, epochs=200, lr=0.001, factor=0.1, patience=10):
        """
        Trains the regression model.

        Args:
            x (torch.tensor):       Input tensor.
            y (torch.tensor):       Target tensor.
            epochs (int, optional): Number of epochs to train.
            lr (float, optional):   Learning rate for the optimizer.

        This method trains the model using a torch optimizer.
        """
        optimizer = torch.optim.Adam(self.params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)[:,0,0].unsqueeze(1) # extract sigma_1_1
            loss   = torch.nn.MSELoss()(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
'''


    def _write_brain_shear_fit_code(self):
        return '''
    def fit(self, x, y, epochs=200, lr=0.001, factor=0.1, patience=10):
        """
        Trains the regression model.

        Args:
            x (torch.tensor):       Input tensor.
            y (torch.tensor):       Target tensor.
            epochs (int, optional): Number of epochs to train.
            lr (float, optional):   Learning rate for the optimizer.

        This method trains the model using a torch optimizer.
        """
        optimizer = torch.optim.Adam(self.params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)[:,0,1].unsqueeze(1) # extract sigma_1_2
            loss   = torch.nn.MSELoss()(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
'''
