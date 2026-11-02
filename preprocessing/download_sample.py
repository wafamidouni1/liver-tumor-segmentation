import urllib.request, tarfile, os

output_dir = r"./data\images"
os.makedirs(output_dir, exist_ok=True)

# Already have these
existing = set(os.listdir(output_dir))
print(f"Already have: {existing}")

url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
print("Downloading new cases from Task03...")

new_count = 0
target    = 3  # download 3 new cases

with urllib.request.urlopen(url) as response:
    with tarfile.open(fileobj=response,
                      mode='r|') as tar:
        for member in tar:
            filename = os.path.basename(member.name)
            if filename.startswith('._'):
                continue
            if 'imagesTr' in member.name and \
               member.name.endswith('.nii.gz') and \
               filename not in existing and \
               new_count < target:
                member.name = filename
                tar.extract(member, output_dir)
                existing.add(filename)
                new_count += 1
                print(f"✓ New case {new_count}/{target}:"
                      f" {filename}")
            if new_count >= target:
                print(f"✓ Got {target} new cases!")
                break

print(f"\nTotal cases: {len(os.listdir(output_dir))}")