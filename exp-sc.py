from manim import *
import numpy as np

# This custom VGroup will represent our numerical vectors without needing LaTeX.
class NumericalVector(VGroup):
    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        number_mobs = VGroup(*[
            Text(f"{val:.2f}", font_size=24) for val in values
        ]).arrange(DOWN, buff=0.15)
        brackets = Brace(number_mobs, direction=LEFT), Brace(number_mobs, direction=RIGHT)
        self.add(number_mobs, *brackets)

# The final, combined scene for a single render.
class TokenizationAndEmbeddings(ThreeDScene):
    def construct(self):
        # --- Part 1: Tokenization (2D Part) ---
        self.play_tokenization_section()
        self.wait(2)
        
        # --- Part 2: Embedding Space (3D Part) ---
        self.play_embedding_space_section()
        self.wait(5)
        
        # --- Part 3: Semantic Arithmetic (3D Part) ---
        self.play_semantic_arithmetic_section()
        self.wait(4)

    def play_tokenization_section(self):
        # Move the camera to a 2D view for this section
        self.set_camera_orientation(phi=0, theta=-90*DEGREES)
        
        title = Text("Tokenization", font_size=72, weight=BOLD).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Create separate Text mobjects for each word and arrange them.
        sentence_words = ["The", "cat", "sat", "on", "the", "mat"]
        words = VGroup(*[Text(word, font_size=48) for word in sentence_words])
        words.arrange(RIGHT, buff=0.4)
        words.next_to(title, DOWN, buff=1.0)

        self.play(FadeIn(words))
        self.wait(1)

        boxes = VGroup(*[SurroundingRectangle(word, buff=0.1, corner_radius=0.1) for word in words])
        self.play(LaggedStart(*[Create(box) for box in boxes], lag_ratio=0.2))
        self.wait(2)

        sentence_group = VGroup(words, boxes)
        self.play(sentence_group.animate.shift(UP * 1.5))
        
        running_word = Text("running", font_size=48).next_to(sentence_group, DOWN, buff=1.5)
        self.play(FadeIn(running_word))
        self.wait(1)

        run_part = Text("run", font_size=48).move_to(running_word.get_center()).shift(LEFT*0.5)
        ing_part = Text("##ing", font_size=48).next_to(run_part, RIGHT, buff=0.05)
        run_box = SurroundingRectangle(run_part, buff=0.1, corner_radius=0.1)
        ing_box = SurroundingRectangle(ing_part, buff=0.1, corner_radius=0.1)
        
        self.play(Transform(running_word, VGroup(run_part, ing_part)), Create(run_box), Create(ing_box))
        self.wait(3)

        # Fade out all 2D elements before transitioning to 3D
        self.play(FadeOut(title, sentence_group, running_word, run_box, ing_box))

    def play_embedding_space_section(self):
        # Now we start the 3D animations
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes()
        self.play(Create(axes))

        locations = {
            "king": np.array([2, 2, 2]), "queen": np.array([2.5, 2.5, 2.5]),
            "man": np.array([2.2, 0, 0]), "woman": np.array([2.7, 0.5, 0.5]),
            "table": np.array([-3, -3, -3]),
        }
        
        # Transition from 2D words to 3D points
        words_2d = VGroup(
            Text("king", font_size=48),
            Text("queen", font_size=48),
            Text("man", font_size=48),
            Text("woman", font_size=48),
            Text("table", font_size=48)
        ).arrange(DOWN, buff=0.5).to_corner(UL)

        self.add_fixed_in_frame_mobjects(words_2d)
        self.play(FadeIn(words_2d))
        self.wait(1)
        
        dots_and_labels = VGroup()
        for i, (word, pos) in enumerate(locations.items()):
            dot = Dot3D(point=pos, color=YELLOW, radius=0.1)
            label = Text(word, font_size=24).next_to(dot, OUT, buff=0.2)
            self.add_fixed_orientation_mobjects(label)
            dots_and_labels.add(dot, label)
            self.play(Transform(words_2d[i].copy(), VGroup(dot, label)), run_time=0.5)

        self.play(FadeOut(words_2d))
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(*self.mobjects))

    def play_semantic_arithmetic_section(self):
        self.move_camera(phi=75 * DEGREES, theta=-30 * DEGREES)
        axes = ThreeDAxes()
        self.play(Create(axes))

        locations = {
            "king": np.array([2, 2, 0]), "queen": np.array([0, 4, 0]),
            "man": np.array([3, 0, 0]), "woman": np.array([1, 2, 0]),
        }

        vectors_and_labels = VGroup()
        for word, pos in locations.items():
            label = Text(word, font_size=36).next_to(pos, OUT, buff=0.3)
            vec = Arrow3D(start=ORIGIN, end=pos, color=BLUE, resolution=8)
            self.add_fixed_orientation_mobjects(label)
            vectors_and_labels.add(vec, label)
            
        self.play(LaggedStart(*[Create(mob) for mob in vectors_and_labels], lag_ratio=0.15))
        self.wait(1)

        # Fixed: Use Create instead of GrowArrow for Arrow3D
        gender_vector = Arrow3D(start=locations["man"], end=locations["woman"], color=GREEN, resolution=8)
        self.play(Create(gender_vector))
        self.wait(1)
        
        moved_gender_vector = gender_vector.copy().shift(locations["king"] - locations["man"])
        self.play(Transform(gender_vector, moved_gender_vector))
        self.wait(1)

        queen_dot = Dot3D(point=locations["queen"], color=YELLOW, radius=0.15)
        self.play(FadeIn(queen_dot, scale=5))
        self.play(queen_dot.animate.set_color(WHITE), run_time=2, rate_func=there_and_back)
        self.wait(2)
        
        equation = Text("king - man + woman ≈ queen", font_size=48).to_corner(UL, buff=0.5)
        equation_bg = SurroundingRectangle(equation, buff=0.1, fill_color=BLACK, fill_opacity=0.7, stroke_width=0)
        self.add_fixed_in_frame_mobjects(equation_bg, equation)
        self.play(Write(equation))
