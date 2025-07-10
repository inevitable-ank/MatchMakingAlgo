import json

# Read all profile chunks
with open('profiles_25.json', 'r') as f:
    profiles_25 = json.load(f)

with open('profiles_50.json', 'r') as f:
    profiles_50 = json.load(f)

with open('profiles_75.json', 'r') as f:
    profiles_75 = json.load(f)

with open('profiles_100.json', 'r') as f:
    profiles_100 = json.load(f)

# Combine all profiles
all_profiles = profiles_25 + profiles_50 + profiles_75 + profiles_100

# Write combined profiles to main profiles.json
with open('profiles.json', 'w') as f:
    json.dump(all_profiles, f, indent=2)

print(f"Successfully combined {len(all_profiles)} profiles into profiles.json") 