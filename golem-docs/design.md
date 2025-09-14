the python library of SURUS AI will have the support to make requests to each AI model type like audio, text, vision, and embeddings models, just like any other package like that of Together AI, OpenAI, anthropic, replicate, and so on. 

But SURUS vision is that of "Task-oriented" AI, where we see AI models as **learned** (instead of written) programs,and programs perform **tasks**. And one of the big revolutions is that of effectively and scalably learning programs that we don't know how to write ourselves. And these learned programs can be modified in two ways: 1. that of software 2.0, where you change the params of the model or the whole model itself, and 3. that of software 3.0, where you change the prompt to modify the behaviour during inference. 

So for example, we aim to have `surus.transcribe()` where we choose the default transcription model and add a "prompt module" to further adapt the behaviour of the program. 

As an important UX decision, we believe that the user just wants to solve a task (like summarize, transcribe, extract_to_json, chat, annotate, and so on), and they don't care about the model behind it, and would love to not have to optimize a prompt. They just want a task solved, and all the companies integrating AI are in some way using the new programs to solve tasks that they couldn't solve before, like good transcription or good document parsing to json. Nothing more, no AI things, no AGI. And we think each verb should have a `high_performance=False` param, where they can change it to `True` and get better performance at a higher price because it switchs to a more powerful but expensive model. This is the only decision a user should make at the beggning: "do you want to solve the task with the good model, or you want the highest perf with the best model?" Also this limits our AI infra to just 2 models per task. And if they need more, we progressively disclousure more complexity of each verb, allowing more control using low level params, up to controlling the model on behind that task and the prompt module adapting the behaviour. They should be able to compose the base prompt module by appending more things, and if necessary remove the entire default prompt module and switch if with an entirely new one made from scratch. 


One thing to assume is that eventually there will be a garden of good prompts that are optimized, and that will have versions or adaptations between different model backends. 
Or something like a Prompt Module Gallery. The best way to develop a prompt 


And the way to make this library has to be highly pragmatic and composable. We have to be able to develop the basics in 1 day, and then progressively add support for new verbs, with different models as backend, and different prompt modules. 
