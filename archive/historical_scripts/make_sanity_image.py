from PIL import Image, ImageDraw

W, H = 768, 512
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# 5 red circles
red_centers = [
    (110, 120),
    (250, 110),
    (390, 125),
    (530, 115),
    (660, 130),
]
r = 35

for x, y in red_centers:
    draw.ellipse(
        [x-r, y-r, x+r, y+r],
        fill="red",
        outline="black",
        width=3,
    )

# 3 blue squares
blue_centers = [
    (180, 330),
    (380, 335),
    (590, 325),
]
s = 60

for x, y in blue_centers:
    draw.rectangle(
        [x-s//2, y-s//2, x+s//2, y+s//2],
        fill="blue",
        outline="black",
        width=3,
    )

img.save("data/sanity_count.png")

print("Saved: data/sanity_count.png")
print("Ground truth:")
print("  red circles = 5")
print("  blue squares = 3")
