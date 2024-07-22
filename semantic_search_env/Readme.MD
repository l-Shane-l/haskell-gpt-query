# Notes

- This is a failed experiment. The idea was to use a vector database and semantic search to send relevant code with a request to gpt to get better anwsers

- I wanted to use this with the refelx libraries in Haskell to mine them for relevant code.

- I also provided an interface to gpt in which it could query the database for more information or attempt to code. 

- When it coded the code would be placed in a template project automatically and all build errors fed back until the code compiled. 

# Hopes

- The hope was I could use this to generate sample widgets to use in reflex like a click and drag list, or even figure out console logging.

# Results

- In the end these tasks proved way too compilcated for the gpt 4o, the time spent getting working code was much greater than the time needed to just figure it out myself.

- Worse than this the expense was way to much for what it is, the code did build but it wasnt of any quality and rarely achieved the goal but instead was more like a random generator, 

- But what really soured my view towards gpt was the limitation in how much information could be sent. It was a constant fight to stay under this.

- Worse than the limit was how inefficient the model was at extracting patterns, I often shared two examples of working code and requested a combination, or sent a chunk of code with a small demonstration example, in any case the model seemed totall unable to learn from what I shared with it.

# Future

- This project is dead for now, I am going to try and limited my use of gpt, it is always tempting because it promises to qucikly solve the issue. 
  but experience shows me it always takes longer and is less satifactory than just doing it myself.

