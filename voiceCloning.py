import outetts

# Configure the model
model_config = outetts.HFModelConfig_v1(
    model_path="OuteAI/OuteTTS-0.2-500M",
    language="en",  # Supported languages in v0.2: en, zh, ja, ko
)

# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.2", cfg=model_config)

# Print available default speakers
interface.print_default_speakers()

# # Load a default speaker
# speaker = interface.load_default_speaker(name="male_3")

#custom speaker
speaker = interface.create_speaker(
    audio_path="RickFinal.mp3",

    # If transcript is not provided, it will be automatically transcribed using Whisper
    transcript="jerry it's your house whatever you say it is is how it is but i think a blind man can see that beth is looking for the i barely a reason to it break the cycle morty rise above focus on science",            # Set to None to use Whisper for transcription

    whisper_model="turbo",      # Optional: specify Whisper model (default: "turbo")
    whisper_device=None,        # Optional: specify device for Whisper (default: None)
)

# Generate speech
output = interface.generate(
    text= """
'Day 1: Foundations and Basics of BERT', 'Day 2: Generalizing to New Tasks with BERT', 'Day 3: Fine-Tuning BERT for Specific Domains', 'Day 4: Advanced Techniques and Tools for BERT', 'Day 5: Practical Applications and Case Studies with BERT', 'Day 6: Reviewing and Refining BERT for Improved Accuracy'
"""
,
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,

    # Optional: Use a speaker profile for consistent voice characteristics
    # Without a speaker profile, the model will generate a voice with random characteristics
    speaker=speaker,
)




# Save the generated speech to a file
output.save("output.wav")