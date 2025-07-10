import core.optimizers   as O
import torch
import core.losses as l
import core.metrics as M
class Deep_learning_Model:

    def __init__(self,optimizer,loss):
        self.layers=[]
        self.weights=[]
        if optimizer=="gradient_descent":
            optimzer=O.GradientDescent()
        elif optimizer=="momentum":
            optimzer=O.Momentum()
        elif optimizer=="adagrad":
            optimzer=O.Adagrad()
        elif optimizer=="rmsprop":
            optimzer=O.RMSProp()
        elif optimizer=="adam":
            optimzer=O.Adam()
        if loss=="Mse":
            loss =l.MSE()
        elif loss=="Mae":
            loss=l.MAE()
        elif loss=='Crossentropy':
            loss=l.CrossEntropy()
        elif loss=='BinaryCrossentropy':
            loss=l.BinaryCrossEntropy()
        self.optimizer=optimzer
        self.loss=loss

    def forward_propagation(self,x):
        y=x
        for layer in self.layers:
            y=layer.forward(y)
        return y
    def backward_propagation(self,lr,t):
        for w in self.weights:
                w.data=self.optimizer.update(w,lr,t)
        for w in self.weights:
            w.grad.zero_()

    def batch_gd_train(self, epochs, x_train, y_train, x_val, y_val,learning_rate,t,weight_decay=False,early_stopping=False,patience=None):
        
        losses, val_losses = [], []
        best_loss = float('inf')
        counter=0
        for epoch in range(epochs):
            train_pred = self.forward_propagation(x_train)
            loss = (train_pred, y_train)
            
            loss.backward()
            if isinstance(self.optimizer,O.Adam):
                 self.backward_propagation(learning_rate,epoch+1)
            else:
                 self.backward_propagation(learning_rate)            
            self.backward_propagation(learning_rate,epoch+1)

            with torch.zero_grad():
                val_pred = self.forward(x_val)
                val_loss = (val_pred, y_val).item()
            losses.append(loss.item())
            val_losses.append(val_loss)
            if early_stopping==True:
             if val_loss < best_loss:
              best_loss = val_loss
              counter = 0  # Reset patience counter
            else:
              counter += 1

        # Early stopping condition
            if counter > patience:
             print(f"Early stopping at epoch {epoch+1}")
             return losses, val_losses 
            print(f"{epoch+1} : train_loss: {loss}  | val_loss:{val_loss}") 
        return losses, val_losses


    # Mini-Batch Stochastic Gradient Descent
    def minibatch_SGD_train(self, epochs, x_train, y_train, x_val, y_val, batch_size,learning_rate,accuracy=False,weight_decay=False,early_stopping=False, patience=None):
        losses, val_losses = [], []
        accuracys,val_accuracys=[],[]
        num_batches = len(x_train) // batch_size
        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            indices = torch.randperm(len(x_train))
            x_train = x_train[indices]
            y_train = y_train[indices]

            for i in range(num_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                x_batch, y_batch = x_train[start:end], y_train[start:end]
                train_pred = self.forward_propagation(x_batch)
                if accuracy==True:
                    softmax_train_scores=l.CrossEntropy.softmax(None,train_pred)
                    print(softmax_train_scores.shape)
                    print(y_batch.shape)
                    acc=M.accuracy(y_batch,softmax_train_scores)
                    

                loss = self.loss.compute_loss(train_pred, y_batch)
                
                loss.backward()
                if isinstance(self.optimizer,O.Adam):
                 self.backward_propagation(learning_rate,epoch+1)
                else:
                 self.backward_propagation(learning_rate)

            # Validation
            val_pred = self.forward_propagation(x_val)
            val_loss = self.loss.compute_loss(val_pred, y_val).item()
            accuracys.append(acc)
            losses.append(loss.item())
            val_losses.append(val_loss)
            if accuracy==True:
                softmax_val_socres=l.CrossEntropy.softmax(None,val_pred)

                val_acc=M.accuracy(y_val,softmax_val_socres)

                val_accuracys.append(val_acc)
            # Early stopping
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0  # reset patience counter
                else:
                    counter += 1

                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    return losses, val_losses
            print(f"{epoch+1} : train_loss: {loss}  | val_loss:{val_loss}")
            
        if accuracy==True:
         print(f"{epoch+1} : train accracy : {acc}  | val accuracy:{val_acc}")
         return losses, val_losses,accuracys,val_accuracys
        else:
         return losses, val_losses


    def SGD_train(self, epochs, x_train, y_train, x_val, y_val,learning_rate, weight_decay=False,early_stopping=False, patience=None):
        losses, val_losses = [], []
        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            indices = torch.randperm(len(x_train))
            x_train = x_train[indices]
            y_train = y_train[indices]

            for i in range(len(x_train)):
                xi, yi = x_train[i].unsqueeze(0), y_train[i].unsqueeze(0)
                train_pred = self(xi)
                loss = self.loss.compute_loss(train_pred, yi)
                loss.backward()
                if isinstance(self.optimizer,O.Adam):
                 self.backward_propagation(learning_rate,epoch+1)
                else:
                 self.backward_propagation(learning_rate)
                self.backward_propagation(loss,epoch+1)

            # Validation
            val_pred = self.forward_propagation(x_val)
            val_loss = self.loss.compute_loss(val_pred, y_val).item()

            losses.append(loss.item())
            val_losses.append(val_loss)

            # Early stopping
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1

                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    return losses, val_losses
            print(f"{epoch+1} : train_loss: {loss}  | val_loss:{val_loss}")
        return losses, val_losses


            

            
                
        
        




    
