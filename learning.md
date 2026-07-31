# Deformable Parts Model (DPM)

**Idea:** An object is not one rigid blob — it has parts that can move a little relative to each other (like a person: head, torso, legs).

**How it works (simple):**
1. Slide a window across the image (check every location / scale).
2. At each spot, look for a “root” (the whole object) plus smaller part detectors (head, arms, etc.).
3. Allow those parts to shift a bit from their usual places — that’s the “deformable” part.
4. Score = how well the root matches + how well the parts match − a small penalty if parts are too far from where they usually sit.

**Takeaway:** Classic pre-deep-learning detector. Good intuition for “objects = parts + flexible layout,” but slow and weaker than modern CNNs.

---

# R-CNN (Region-Based CNN)

**Idea:** Don’t scan every pixel with a CNN. First propose a few likely object boxes, then classify each box.

**How it works (simple):**
1. **Region proposals** — an algorithm (Selective Search) suggests ~2000 boxes that might contain objects.
2. **Warp** each box to a fixed size and run it through a CNN to get a feature vector.
3. **Classify** that vector with an SVM (is it a cat? car? background?).
4. **Refine** the box coordinates a bit (bounding-box regression).

**Takeaway:** Big leap for accuracy by using deep features, but slow (CNN runs once per box). Later versions — Fast R-CNN, Faster R-CNN — reuse computation and make proposals smarter so detection gets much faster

# YOLO (YOLOv1)

**Idea:** Treat detection as one shot — one CNN pass predicts all boxes and classes at once. No region proposals.

**How it works (simple):**
1. Split the image into an **S×S grid** (e.g. 7×7).
2. Each cell is responsible for objects whose **center** falls in that cell.
3. For each cell the network predicts:
   - **Class probabilities** — “what is it?” (dog, car, …) for that cell
   - **B bounding boxes**, each with `(x, y, w, h)` + a **confidence** score (“is there an object, and how tight is the box?”)
4. Confidence × class score → final “this is a dog here” score. Weak boxes get dropped (NMS).

**Pipeline:** One CNN (stacked conv layers) → feature map → final tensor of size roughly `S × S × (B·5 + C)` (5 = x,y,w,h,confidence; C = classes).

**Takeaway:** Fast because it’s one forward pass. Weaker than two-stage detectors on small/overlapping objects in v1, but later YOLO versions fix a lot of that.