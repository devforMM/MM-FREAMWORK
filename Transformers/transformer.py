from transformer_operations import *
from Heads import *
from transformer_operations import *
class transformer_encoder:
    def __init__(self,num_heads,d_model):
        self.encoder_weights=[]
        self.num_heads=num_heads
        self.norm_layer=AddNorm_Layer(d_model)
        self.Feed_forward=Feed_Forward(d_model)
        self.mulit_head_attention_layer=Multi_head_attention_layer(d_model,num_heads)
        self.norm_layer2=AddNorm_Layer(d_model)
        self.encoder_weights.extend(self.mulit_head_attention_layer.weights)
        self.encoder_weights.extend(self.norm_layer.weights)
        self.encoder_weights.extend(self.Feed_forward.weights)
        self.encoder_weights.extend(self.norm_layer2.weights)
    def encode(self,source_embedings):
            scores=self.mulit_head_attention_layer.forward(source_embedings)
            norm_scores=self.norm_layer.forward(source_embedings,scores)
            FFn_scores=self.Feed_forward.forward(norm_scores)
            norm_Fnn_socres=self.norm_layer2.forward(norm_scores,FFn_scores)
            return norm_Fnn_socres
    
class transformer_decoder:
    def __init__(self,num_heads,d_model,vocab_size):
        self.decoder_weights=[]
        self.num_heads=num_heads
        self.norm_layer=AddNorm_Layer(d_model)
        self.norm_layer2=AddNorm_Layer(d_model)
        self.norm_layer3=AddNorm_Layer(d_model)
        self.Feed_forward=Feed_Forward(d_model)
        self.linear=Linear(d_model,vocab_size)
        self.masked_multi_head_attention_layer=masked_Multi_head_attention_layer(d_model,num_heads)
        self.cross_multi_head_attention_layer=cross_Multi_head_attention_layer(d_model,num_heads)
        self.decoder_weights.extend(self.masked_multi_head_attention_layer.weights)
        self.decoder_weights.extend(self.norm_layer.weights)
        self.decoder_weights.extend(self.cross_multi_head_attention_layer.weights)
        self.decoder_weights.extend(self.norm_layer2.weights)
        self.decoder_weights.extend(self.Feed_forward.weights)
        self.decoder_weights.extend(self.norm_layer3.weights)
        
    def decode(self, encoder_scores, target_embeddings):
        self.decoder_weights.extend(self.linear.weights)
        masked_scores = self.masked_multi_head_attention_layer.forward(target_embeddings)
        norm_masked_scores = self.norm_layer.forward(target_embeddings, masked_scores)
        cross_scores = self.cross_multi_head_attention_layer.forward(encoder_scores, norm_masked_scores)
        norm_multi_scores = self.norm_layer2.forward(norm_masked_scores, cross_scores)
        FFN_scores = self.Feed_forward.forward(norm_multi_scores)
        final_norm_scores = self.norm_layer3.forward(norm_multi_scores, FFN_scores)
        softmax_scores = self.linear.forward(final_norm_scores)
        return softmax_scores
    def  multi_layer_decode(self, encoder_scores, target_embeddings):
        masked_scores = self.masked_multi_head_attention_layer.forward(target_embeddings)
        norm_masked_scores = self.norm_layer.forward(target_embeddings, masked_scores)
        cross_scores = self.cross_multi_head_attention_layer.forward(encoder_scores, norm_masked_scores)
        norm_multi_scores = self.norm_layer2.forward(norm_masked_scores, cross_scores)
        FFN_scores = self.Feed_forward.forward(norm_multi_scores)
        final_norm_scores = self.norm_layer3.forward(norm_multi_scores, FFN_scores)
        return final_norm_scores
         

class  transformer_bloc:
     def __init__(self,num_heads,dmodel,vocab_size):
          self.dmodel=dmodel
          self.Encoder=transformer_encoder(num_heads,dmodel)
          self.Decoder=transformer_decoder(num_heads,dmodel,vocab_size)
          self.w=self.Encoder.encoder_weights+self.Decoder.decoder_weights
     def forward(self,source,targets):
          pos_source=pos_encoding(source,self.dmodel)
          encoder_scores=self.Encoder.encode(pos_source)
          pos_targets=pos_encoding(targets,self.dmodel)
          decoder_scores=self.Decoder.decode(encoder_scores,pos_targets)
          return decoder_scores



