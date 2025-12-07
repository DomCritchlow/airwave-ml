"""
Attention-based Seq2Seq model for audio-to-text.
Uses Transformer encoder-decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class AudioEncoder(nn.Module):
    """Encoder for audio features."""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Project input features to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Conv layers for local feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.d_model = d_model
    
    def forward(self, src, src_lengths=None):
        """
        Args:
            src: (batch, time, freq)
            src_lengths: (batch,) lengths of each sequence
            
        Returns:
            encoded: (batch, time, d_model)
        """
        # Project to d_model
        x = self.input_projection(src)  # (batch, time, d_model)
        
        # Apply conv layers
        x_conv = x.transpose(1, 2)  # (batch, d_model, time)
        x_conv = self.conv(x_conv)
        x = x_conv.transpose(1, 2)  # (batch, time, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if lengths provided
        mask = None
        if src_lengths is not None:
            batch_size, max_len = x.size(0), x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= src_lengths.unsqueeze(1)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return encoded


class AudioToTextModel(nn.Module):
    """Complete attention-based audio-to-text model."""
    
    def __init__(self, vocab_size, input_dim, config):
        super().__init__()
        
        d_model = config['model']['d_model']
        nhead = config['model']['nhead']
        num_encoder_layers = config['model']['num_encoder_layers']
        num_decoder_layers = config['model']['num_decoder_layers']
        dim_feedforward = config['model']['dim_feedforward']
        dropout = config['model']['dropout']
        
        # Audio encoder
        self.encoder = AudioEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Text decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_lengths, tgt, tgt_lengths=None):
        """
        Args:
            src: (batch, src_time, freq)
            src_lengths: (batch,)
            tgt: (batch, tgt_time) - target sequence
            tgt_lengths: (batch,) - optional
            
        Returns:
            output: (batch, tgt_time, vocab_size)
        """
        # Encode audio
        memory = self.encoder(src, src_lengths)  # (batch, src_time, d_model)
        
        # Embed target sequence
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_decoder(tgt_embedded)
        
        # Create masks
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Padding masks
        tgt_padding_mask = None
        if tgt_lengths is not None:
            tgt_padding_mask = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0) >= tgt_lengths.unsqueeze(1)
        
        memory_padding_mask = None
        if src_lengths is not None:
            memory_padding_mask = torch.arange(memory.size(1), device=memory.device).unsqueeze(0) >= src_lengths.unsqueeze(1)
        
        # Decode
        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder self-attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def greedy_decode(self, src, src_lengths, max_len=200, sos_token=1, eos_token=2):
        """
        Greedy decoding for inference.
        
        Args:
            src: (1, src_time, freq) - single audio sample
            src_lengths: (1,)
            max_len: maximum generation length
            sos_token: start of sequence token
            eos_token: end of sequence token
            
        Returns:
            predicted sequence (list of token indices)
        """
        self.eval()
        with torch.no_grad():
            # Encode
            memory = self.encoder(src, src_lengths)
            
            # Start with SOS token
            ys = torch.ones(1, 1).fill_(sos_token).type_as(src).long()
            
            for i in range(max_len - 1):
                # Embed and decode
                tgt_embedded = self.embedding(ys) * math.sqrt(self.d_model)
                tgt_embedded = self.pos_decoder(tgt_embedded)
                
                tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(src.device)
                
                out = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
                out = self.fc_out(out)
                
                # Get next token
                prob = out[:, -1, :]
                next_word = prob.argmax(dim=-1).item()
                
                # Append to sequence
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src).long().fill_(next_word)], dim=1)
                
                # Stop if EOS
                if next_word == eos_token:
                    break
            
            return ys.squeeze(0).tolist()
    
    def beam_search_decode(self, src, src_lengths, beam_width=5, max_len=200, sos_token=1, eos_token=2):
        """
        Beam search decoding for better inference.
        
        Args:
            src: (1, src_time, freq)
            src_lengths: (1,)
            beam_width: number of beams
            max_len: maximum generation length
            sos_token: start token
            eos_token: end token
            
        Returns:
            best predicted sequence
        """
        self.eval()
        with torch.no_grad():
            # Encode
            memory = self.encoder(src, src_lengths)  # (1, src_time, d_model)
            
            # Initialize beams
            beams = [([sos_token], 0.0)]  # (sequence, score)
            
            for _ in range(max_len):
                candidates = []
                
                for seq, score in beams:
                    if seq[-1] == eos_token:
                        candidates.append((seq, score))
                        continue
                    
                    # Prepare input
                    ys = torch.tensor([seq], dtype=torch.long, device=src.device)
                    
                    # Decode
                    tgt_embedded = self.embedding(ys) * math.sqrt(self.d_model)
                    tgt_embedded = self.pos_decoder(tgt_embedded)
                    tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(src.device)
                    
                    out = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
                    out = self.fc_out(out)
                    
                    # Get top k tokens
                    log_probs = F.log_softmax(out[:, -1, :], dim=-1)
                    topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                    
                    for prob, idx in zip(topk_probs[0], topk_indices[0]):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        candidates.append((new_seq, new_score))
                
                # Select top beams
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Check if all beams ended
                if all(seq[-1] == eos_token for seq, _ in beams):
                    break
            
            # Return best sequence
            best_seq, _ = beams[0]
            return best_seq


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
