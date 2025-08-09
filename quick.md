# Manim Python Library: Syntax, Terms, and Examples



## Manim Concepts, Syntax & Example Table

| Term/Syntax | Meaning | Short Explanation | Example |
|---|---|---|---|
| `from manim import *` | Import Manim | Loads all Manim classes and functions for use in your script | `from manim import *` |
| `Scene` | Base Animation Class | Central class; every animation is a subclass of Scene | `class MyScene(Scene): ...` |
| `construct()` | Main Animation Method | All animation code goes inside this method, called automatically | `def construct(self): ...` |
| `Circle()`, `Square()`, etc. | Shape Objects | Built-in shapes ("Mobjects") for animation | `circle = Circle()` |
| `.set_fill(COLOR, opacity=...)` | Set Color & Opacity | Fill color/opacity of shape | `circle.set_fill(PINK, opacity=0.5)` |
| `.play()` | Play Animation | Triggers animation (draw, transform, etc.) | `self.play(Create(circle))` |
| `Create()`, `Write()`, etc. | Animation Functions | Built-in animation types you can apply to Mobjects | `self.play(Write(text))` |
| `Text()`, `MathTex()` | Text Display | Display regular text and LaTeX math | `text = Text("Hello")` |
| `Line()`, `Arrow()` | Drawing Lines/Arrows | Draw lines, arrows, vectors in scenes | `arrow = Arrow(ORIGIN, [1][1])` |
| `VGroup()` | Group Objects | Group multiple Mobjects for animation | `group = VGroup(circle, square)` |
| `self.add(obj)` | Add to Scene | Instantly shows (no animation) | `self.add(text)` |
| `.animate` | Animate Properties | Change properties with animation (position, color, etc.) | `self.play(circle.animate.shift(RIGHT))` |
| `NumberPlane()` | Coordinate Plane | Display a grid for graphs/vectors | `plane = NumberPlane()` |
| `Axes()` | Axes System | Create x-y axes for plotting functions | `axes = Axes(...)` |
| `.plot()` | Plot Functions | Plot mathematical functions | `sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)` |
| `.wait()` | Pause Animation | Wait for specified seconds in the animation | `self.wait(2)` |

***



For more complex examples and further learning, the [Manim documentation](https://docs.manim.community/en/stable/tutorials/quickstart.html)
