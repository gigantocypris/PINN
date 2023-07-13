"""Tests for PINN"""

# Standard library imports
import pathlib

# Third party imports
import pytest


"""
Tests to write:
Scattered field is zero when there is no scatterer
Scattered field is nonzero when there is a scatterer and the incident field is nonzero
Scattered field is zero when there is a scatterer and the incident field is zero
Scattered field matches analytic solution for simple geometries
Distributed code matches results of non-distributed code
"""

PUZZLE_DIR = pathlib.Path(__file__).parent


@pytest.fixture
def example1():
    puzzle_input = (PUZZLE_DIR / "example1.txt").read_text().strip()
    return aoc202201.parse_data(puzzle_input)


@pytest.fixture
def example2():
    puzzle_input = (PUZZLE_DIR / "example2.txt").read_text().strip()
    return aoc202201.parse_data(puzzle_input)


def test_parse_example1(example1):
    """Test that input is parsed properly."""
    assert example1 == [[1001],[1002, 1003],[1003, 1004]]

def test_parse_example2(example2):
    """Test that input is parsed properly."""
    assert example2 == [[1000, 1500],[2000, 2000]]

def test_part1_example1(example1):
    """Test part 1 on example input."""
    assert aoc202201.part1(example1) == 2007

def test_part1_example2(example2):
    """Test part 1 on example input."""
    assert aoc202201.part1(example2) == 4000
    
def test_part2_example1(example1):
    """Test part 2 on example input."""
    assert aoc202201.part2(example1) == 5013

def test_part2_example2(example2):
    """Test part 2 on example input."""
    assert aoc202201.part2(example2) == 6500
