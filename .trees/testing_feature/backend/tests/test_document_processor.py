import pytest
import tempfile
import os
from document_processor import DocumentProcessor
from models import Course, Lesson, CourseChunk


class TestDocumentProcessor:
    """Test cases for DocumentProcessor"""

    def test_initialization(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        assert processor.chunk_size == 800
        assert processor.chunk_overlap == 100

    def test_read_file_utf8(self, temp_dir):
        """Test reading UTF-8 encoded files"""
        # Create test file
        test_file = os.path.join(temp_dir, "test.txt")
        content = "Test content with UTF-8 characters: café, résumé, naïve"

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)

        processor = DocumentProcessor(100, 20)
        result = processor.read_file(test_file)
        assert result == content

    def test_read_file_with_encoding_errors(self, temp_dir):
        """Test reading files with encoding issues"""
        test_file = os.path.join(temp_dir, "test.txt")

        # Write binary data that might cause encoding issues
        with open(test_file, 'wb') as f:
            f.write(b"Normal text\xff\xfe and some problematic bytes")

        processor = DocumentProcessor(100, 20)
        # Should not raise an exception, should handle gracefully
        result = processor.read_file(test_file)
        assert "Normal text" in result

    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = processor.chunk_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some flexibility

    def test_chunk_text_with_short_text(self):
        """Test chunking with text shorter than chunk size"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        text = "Short text."
        chunks = processor.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_overlap_behavior(self):
        """Test that overlap works correctly"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=20)

        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = processor.chunk_text(text)

        # Should have multiple chunks with some overlap
        assert len(chunks) > 1

        # Check that consecutive chunks share some content (overlap)
        if len(chunks) > 1:
            # Find common words between first two chunks
            words1 = set(chunks[0].split())
            words2 = set(chunks[1].split())
            common_words = words1.intersection(words2)
            # Should have some overlap
            assert len(common_words) > 0

    def test_chunk_text_whitespace_normalization(self):
        """Test whitespace normalization in chunking"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        text = "Text   with    multiple   spaces\n\nand\tlines."
        chunks = processor.chunk_text(text)

        # Whitespace should be normalized
        assert "   " not in chunks[0]
        assert "\n\n" not in chunks[0]
        assert "\t" not in chunks[0]

    def test_parse_course_document_valid(self, temp_dir, sample_course_document):
        """Test parsing a valid course document"""
        # Write sample document to file
        test_file = os.path.join(temp_dir, "test_course.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(sample_course_document)

        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        course, chunks = processor.process_course_document(test_file)

        # Verify course parsing
        assert course is not None
        assert course.title == "Test Course"
        assert course.instructor == "Test Instructor"
        assert course.course_link == "https://example.com/test-course"
        assert len(course.lessons) == 2

        # Verify lesson parsing
        lesson0 = course.lessons[0]
        assert lesson0.lesson_number == 0
        assert lesson0.title == "Introduction"
        assert lesson0.lesson_link == "https://example.com/lesson0"

        # Verify chunks were created
        assert len(chunks) > 0
        assert all(isinstance(chunk, CourseChunk) for chunk in chunks)
        assert all(chunk.course_title == "Test Course" for chunk in chunks)

    def test_parse_course_document_missing_fields(self, temp_dir):
        """Test parsing document with missing required fields"""
        # Document missing course title
        invalid_document = """Course Link: https://example.com/test
Course Instructor: Test Instructor

Lesson 0: Introduction
Some content here.
"""
        test_file = os.path.join(temp_dir, "invalid.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(invalid_document)

        processor = DocumentProcessor(100, 20)

        # Should handle gracefully or raise informative error
        try:
            course, chunks = processor.process_course_document(test_file)
            # If it doesn't raise an error, should return None or empty data
            assert course is None or course.title is None
        except Exception as e:
            # Should provide informative error message
            assert "title" in str(e).lower() or "course" in str(e).lower()

    def test_parse_course_document_no_lessons(self, temp_dir):
        """Test parsing document with no lessons"""
        document_no_lessons = """Course Title: Test Course
Course Link: https://example.com/test
Course Instructor: Test Instructor

Some general course description without lessons.
"""
        test_file = os.path.join(temp_dir, "no_lessons.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(document_no_lessons)

        processor = DocumentProcessor(100, 20)
        course, chunks = processor.process_course_document(test_file)

        # Should handle documents without lessons
        assert course is not None
        assert len(course.lessons) == 0
        # Might still create chunks from general content

    def test_parse_real_course_document(self, temp_dir):
        """Test parsing a real course document format"""
        # Use the actual format from the docs
        real_document = """Course Title: Building Towards Computer Use with Anthropic
Course Link: https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/
Course Instructor: Colt Steele

Lesson 0: Introduction
Lesson Link: https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction
Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic and taught by Colt Steele, whose Anthropic's Head of Curriculum. Welcome, Colt. Thanks, Andrew. I'm delighted to have the opportunity to share this course with all of you. Anthropic made a recent breakthrough and released a model that could use a computer.

Lesson 1: Getting Started with the API
Lesson Link: https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/abc123/getting-started
In this lesson, you'll learn how to make your first API call to Claude. We'll start with basic text generation and then move on to more advanced features. The Anthropic API is designed to be intuitive and powerful.
"""

        test_file = os.path.join(temp_dir, "real_course.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(real_document)

        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        course, chunks = processor.process_course_document(test_file)

        # Verify parsing of real format
        assert course is not None
        assert course.title == "Building Towards Computer Use with Anthropic"
        assert course.instructor == "Colt Steele"
        assert len(course.lessons) == 2

        # Verify lesson details
        assert course.lessons[0].lesson_number == 0
        assert course.lessons[0].title == "Introduction"
        assert course.lessons[1].lesson_number == 1
        assert course.lessons[1].title == "Getting Started with the API"

        # Should create multiple chunks from content
        assert len(chunks) > 2

        # Verify chunk content
        lesson_0_chunks = [c for c in chunks if c.lesson_number == 0]
        lesson_1_chunks = [c for c in chunks if c.lesson_number == 1]

        assert len(lesson_0_chunks) > 0
        assert len(lesson_1_chunks) > 0

    def test_create_course_chunks(self, temp_dir, sample_course_document):
        """Test chunk creation from parsed course"""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(sample_course_document)

        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        course, chunks = processor.process_course_document(test_file)

        # Verify chunk structure
        for chunk in chunks:
            assert isinstance(chunk, CourseChunk)
            assert chunk.course_title == course.title
            assert chunk.lesson_number in [0, 1]  # From our sample
            assert len(chunk.content) > 0
            assert isinstance(chunk.chunk_index, int)

        # Verify chunk indices are sequential
        chunk_indices = [chunk.chunk_index for chunk in chunks]
        assert chunk_indices == list(range(len(chunks)))

    def test_empty_file_handling(self, temp_dir):
        """Test handling of empty files"""
        test_file = os.path.join(temp_dir, "empty.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("")

        processor = DocumentProcessor(100, 20)

        try:
            course, chunks = processor.process_course_document(test_file)
            # Should handle empty files gracefully
            assert course is None or len(chunks) == 0
        except Exception:
            # If it raises an exception, that's also acceptable for empty files
            pass

    def test_malformed_lesson_format(self, temp_dir):
        """Test handling of malformed lesson formats"""
        malformed_document = """Course Title: Test Course
Course Link: https://example.com/test
Course Instructor: Test Instructor

Lesson ABC: Invalid lesson number
Some content here.

Lesson: Missing lesson number
More content here.

Lesson 1: Valid lesson
Valid content here.
"""
        test_file = os.path.join(temp_dir, "malformed.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(malformed_document)

        processor = DocumentProcessor(100, 20)
        course, chunks = processor.process_course_document(test_file)

        # Should parse what it can and skip malformed lessons
        assert course is not None
        # Should have at least the valid lesson
        valid_lessons = [l for l in course.lessons if isinstance(l.lesson_number, int)]
        assert len(valid_lessons) >= 1

    def test_very_long_content_chunking(self):
        """Test chunking of very long content"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        # Create long content
        long_text = "This is a sentence. " * 100  # 2000+ characters
        chunks = processor.chunk_text(long_text)

        # Should create multiple chunks
        assert len(chunks) > 5

        # Each chunk should be reasonably sized
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some flexibility

        # Should have overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            words1 = set(chunks[i].split())
            words2 = set(chunks[i + 1].split())
            common_words = words1.intersection(words2)
            assert len(common_words) > 0  # Should have overlap

    def test_processor_with_current_system_files(self):
        """Test processor with actual system files if available"""
        # This test checks if the processor can handle the real course files
        docs_path = "../docs"
        if os.path.exists(docs_path):
            processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

            for filename in os.listdir(docs_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(docs_path, filename)
                    try:
                        course, chunks = processor.process_course_document(file_path)

                        # Should successfully parse each file
                        assert course is not None, f"Failed to parse {filename}"
                        assert len(chunks) > 0, f"No chunks created for {filename}"

                        # Verify basic course structure
                        assert course.title is not None
                        assert course.instructor is not None
                        assert len(course.lessons) > 0

                    except Exception as e:
                        pytest.fail(f"Failed to process {filename}: {str(e)}")
        else:
            pytest.skip("Docs directory not available for testing")