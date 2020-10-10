# nohup python main.py --use_syntax --maxpooling --requires_grad --output_dir 'saved_syntax_word' >log.txt &
# python main.py --maxpooling --requires_grad --output_dir 'saved_word' >log.txt &
nohup python main.py --syntax_dim 800  --use_syntax --syntax_hidden_embed --no_gpu 4 --maxpooling --requires_grad --output_dir 'saved_syntax_word_embed_2' >log_embed.txt &