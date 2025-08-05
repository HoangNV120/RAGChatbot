import json
import os
import re
import uuid
from langchain.schema import Document

class SyllabusConverter:
    def __init__(self):
        pass
        
    def is_syllabus_file(self, filename):
        """
        Determine if a file is a syllabus file based on its name
        
        Args:
            filename (str): Filename to check
            
        Returns:
            bool: True if the file is a syllabus, False otherwise
        """
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Pattern for syllabus: 3 uppercase letters (may include Đ) + 3 digits + optional lowercase letter
        syllabus_pattern = r'^[A-ZĐ]{3}\d{3}[a-z]?$'
        
        # Check against pattern
        if re.match(syllabus_pattern, name_without_ext):
            return True
            
        # Check if name contains "LUK"
        if "LUK" in name_without_ext.upper():
            return True
            
        return False
        
    def get_category_parts_info(self, categories):
        """
        Extract information about the number of parts in each category
        
        Args:
            categories (list): List of category dictionaries
            
        Returns:
            dict: Dictionary with category names as keys and part information as values
        """
        parts_info = {}
        
        for category in categories:
            cat_name = category.get("Category", "")
            part = category.get("Part", "1")
            weight = category.get("Weight", "0%")
            
            if cat_name:
                try:
                    part_count = int(part)
                    weight_value = float(weight.replace("%", ""))
                    
                    # Each part has the full weight, and total contribution is weight * parts
                    total_contribution = weight_value * part_count
                    
                    parts_info[cat_name] = {
                        "total_parts": part_count,
                        "per_part_weight": weight_value,  # Each part has the same full weight
                        "total_contribution": total_contribution  # Total contribution to final grade
                    }
                except (ValueError, ZeroDivisionError):
                    parts_info[cat_name] = {
                        "total_parts": 1,
                        "per_part_weight": weight,
                        "total_contribution": weight
                    }
                    
        return parts_info

    def convert_syllabus_to_plain_text(self, json_data):
        """
        Convert course syllabus JSON data to natural language sentences in English.
        Each material, session, assessment (category), and CLO is one sentence.
        Every sentence includes the course code and is written naturally for NLP.

        Args:
            json_data: Parsed JSON data of a course syllabus

        Returns:
            str: Plain text with each item as a single natural sentence containing the course code
        """
        try:
            # Get the course code
            subject_code = json_data.get("Subject Code", "")
            if not subject_code:
                return self._simple_convert(json_data)

            # Extract sentences for each section
            all_sentences = []

            # Basic course information
            all_sentences.extend(self._extract_basic_info(json_data, subject_code))

            # Materials - one sentence per material with full details
            material_sentences = self._extract_materials(json_data.get("Materials", []), subject_code)
            if material_sentences:
                all_sentences.extend(material_sentences)

            # Learning outcomes - one sentence per CLO with full details
            clo_sentences = self._extract_clos(json_data.get("CLOs", []), subject_code)
            if clo_sentences:
                all_sentences.extend(clo_sentences)

            # Sessions - one sentence per session with full details
            session_sentences = self._extract_sessions(json_data.get("StudentMaterials", []), subject_code)
            if session_sentences:
                all_sentences.extend(session_sentences)

            # Assessment categories - one sentence per category with full details
            category_sentences = self._extract_categories(json_data.get("Categories", []), subject_code)
            if category_sentences:
                all_sentences.extend(category_sentences)

            # Combine all sentences into a single document
            all_text = " ".join(all_sentences)

            # Ensure proper spacing after periods
            cleaned_text = re.sub(r'\.(?=[A-Z])', '. ', all_text)

            return cleaned_text

        except Exception as e:
            print(f"Error converting syllabus to plain text: {e}")
            return self._simple_convert(json_data)

    def _extract_basic_info(self, json_data, subject_code):
        """Extract basic course information with each item as one natural sentence containing the course code"""
        sentences = []

        # Syllabus name
        syllabus_name = json_data.get("Syllabus Name", "")
        if syllabus_name:
            sentences.append(f"The course {subject_code} is named {syllabus_name}.")

        # Credits
        credits = json_data.get("NoCredit", "")
        if credits:
            sentences.append(f"The course {subject_code} offers {credits} credits.")

        # Degree level
        degree_level = json_data.get("Degree Level", "")
        if degree_level:
            sentences.append(f"The course {subject_code} is designed for the {degree_level} level.")

        # Time allocation
        time_allocation = json_data.get("Time Allocation", "")
        if time_allocation:
            sentences.append(f"The course {subject_code} allocates {time_allocation} for study.")

        # Pre-requisite
        prerequisite = json_data.get("Pre-Requisite", "")
        if prerequisite:
            sentences.append(f"The course {subject_code} requires completion of {prerequisite} as a prerequisite.")

        # Description (full description)
        description = json_data.get("Description", "")
        if description:
            clean_desc = re.sub(r'\.(?=[A-Z])', '. ', description)
            sentences.append(f"The course {subject_code} teaches {clean_desc}")

        # Student tasks
        student_tasks = json_data.get("StudentTasks", "")
        if student_tasks:
            sentences.append(f"The course {subject_code} expects students to {student_tasks}.")

        return sentences

    def _extract_materials(self, materials, subject_code):
        """Extract course materials with each material as one natural sentence containing the course code"""
        if not materials:
            return []

        sentences = []

        for material in materials:
            description = material.get("MaterialDescription", "")
            if description:
                material_type = "main" if material.get("IsMainMaterial") else "supplementary"
                sentence = f"The course {subject_code} uses {description} as a {material_type} learning material"

                # Add author
                author = material.get("Author", "")
                if author:
                    sentence += f" written by {author}"

                # Add publisher
                publisher = material.get("Publisher", "")
                if publisher:
                    sentence += f" and published by {publisher}"

                # Add published date
                published_date = material.get("PublishedDate", "")
                if published_date:
                    sentence += f" in {published_date}"

                # Add edition
                edition = material.get("Edition", "")
                if edition:
                    sentence += f" in its {edition} edition"

                # Add ISBN
                isbn = material.get("ISBN", "")
                if isbn:
                    sentence += f" with ISBN {isbn}"

                # Add format (hard copy or online)
                format_type = "hard copy" if material.get("IsHardCopy") else "online"
                sentence += f" provided as a {format_type} resource"

                # Add note or URL
                note = material.get("Note", "")
                url = material.get("URL", "")
                if note and note.startswith("http"):
                    sentence += f" and accessible online at {note}"
                elif note:
                    sentence += f" with additional notes indicating {note}"
                elif url:
                    sentence += f" and accessible online at {url}"

                sentence += "."
                sentences.append(sentence)

        return sentences

    def _extract_clos(self, clos, subject_code):
        """Extract course learning outcomes with each CLO as one natural sentence containing the course code"""
        if not clos:
            return []

        sentences = []

        for clo in clos:
            clo_name = clo.get("CLO Name", "")
            lo_details = clo.get("LO Details", "")
            if clo_name and lo_details:
                sentences.append(f"The course {subject_code} includes learning outcome {clo_name} which requires students to {lo_details}.")

        return sentences

    def _extract_sessions(self, sessions, subject_code):
        """Extract course sessions with each session as one natural sentence containing the course code"""
        if not sessions:
            return []

        sentences = []

        for session in sessions:
            session_num = session.get("Session", "")
            topic = session.get("Topic", "")
            if session_num and topic:
                sentence = f"The course {subject_code} in session {session_num} covers {topic}"

                # Add learning-teaching type
                teaching_type = session.get("Learning-Teaching Type", "")
                if teaching_type:
                    sentence += f" delivered through {teaching_type} methods"

                # Add learning outcomes
                lo = session.get("LO", "")
                if lo:
                    sentence += f" to achieve learning outcomes {lo}"

                # Add ITU
                itu = session.get("ITU", "")
                if itu:
                    sentence += f" using instructional types {itu}"

                # Add student tasks
                student_tasks = session.get("Student's Tasks", "")
                if student_tasks:
                    sentence += f" where students are required to {student_tasks}"

                # Add student materials
                student_materials = session.get("Student Materials", "")
                if student_materials:
                    sentence += f" utilizing materials such as {student_materials}"

                # Add URLs
                urls = session.get("URLs", "")
                if urls:
                    sentence += f" with resources available at {urls}"

                sentence += "."
                sentences.append(sentence)

        return sentences

    def _extract_categories(self, categories, subject_code):
        """Extract assessment categories with each category as one natural sentence containing the course code"""
        if not categories:
            return []

        sentences = []

        for category in categories:
            cat_name = category.get("Category", "")
            if cat_name:
                # Start the basic sentence with "grade" term
                sentence = f"The course {subject_code} assesses students through {cat_name} grade"

                # Add weight without multiplying by parts
                weight = category.get("Weight", "")
                part = category.get("Part", "1")

                if weight:
                    sentence += f" which contributes {weight} to the final grade"

                    # Add part information if there are multiple parts
                    try:
                        parts_count = int(part) if part and part != "1" else 1
                        if parts_count > 1:
                            sentence += f" (consisting of {parts_count} parts)"
                    except (ValueError, TypeError):
                        if part != "1":
                            sentence += f" (consisting of {part} parts)"

                # Add type
                type_ = category.get("Type ", "")
                if type_:
                    sentence += f" as a {type_} assessment"

                # Add completion criteria
                completion = category.get("Completion Criteria", "")
                if completion:
                    # Replace > with "Over" for better readability
                    if ">" in completion:
                        completion_text = completion.replace(">", "Over ")
                        sentence += f" requiring passing requirements of {completion_text}"
                    else:
                        sentence += f" requiring passing requirements of {completion}"

                # Add CLOs
                clo = category.get("CLO", "")
                if clo:
                    sentence += f" to evaluate learning outcomes {clo}"

                # Add question type
                question_type = category.get("Question Type", "")
                if question_type:
                    sentence += f" using {question_type}"

                # Add number of questions
                no_question = category.get("No Question", "")
                if no_question:
                    sentence += f" with {no_question} questions"

                # Add knowledge and skill
                knowledge_skill = category.get("Knowledge and Skill", "")
                if knowledge_skill:
                    sentence += f" to assess {knowledge_skill}"

                # Add grading guide
                grading = category.get("Grading Guide", "")
                if grading:
                    sentence += f" conducted during {grading}"

                # Add note
                note = category.get("Note", "")
                if note:
                    sentence += f" with additional notes that {note}"

                sentence += "."
                sentences.append(sentence)

        return sentences

    def _simple_convert(self, json_data):
        """Simple fallback conversion to natural sentences each containing the course code"""
        try:
            subject_code = json_data.get("Subject Code", "UNKNOWN")
            sentences = []

            # Syllabus name
            name = json_data.get("Syllabus Name", "")
            if name:
                sentences.append(f"The course {subject_code} is named {name}.")

            # Description (full)
            description = json_data.get("Description", "")
            if description:
                clean_desc = re.sub(r'\.(?=[A-Z])', '. ', description)
                sentences.append(f"The course {subject_code} teaches {clean_desc}")

            # Other string attributes
            for key, value in json_data.items():
                if key not in ["Subject Code", "Syllabus Name", "Description"] and isinstance(value, str) and value:
                    sentences.append(f"The course {subject_code} includes {key.lower()} of {value}.")

            # Complex structures
            for key, value in json_data.items():
                if isinstance(value, list) and value:
                    sentences.append(f"The course {subject_code} contains {len(value)} {key.lower()}.")

            return " ".join(sentences)

        except Exception as e:
            print(f"Error in simple convert: {e}")
            return f"The course {json_data.get('Subject Code', 'UNKNOWN')} could not be converted due to an error."

    def create_syllabus_summary(self, json_data):
        """
        Create a structured summary of the syllabus with clear information about assessment structure
        
        Args:
            json_data: Parsed JSON data of a course syllabus
            
        Returns:
            str: Formatted summary of the syllabus
        """
        try:
            # Basic course information
            subject_code = json_data.get("Subject Code", "UNKNOWN")
            syllabus_name = json_data.get("Syllabus Name", "")
            credits = json_data.get("NoCredit", "")
            
            summary = []
            summary.append(f"SYLLABUS SUMMARY: {subject_code} - {syllabus_name}")
            summary.append(f"Credits: {credits}")
            summary.append("")
            
            # Assessment structure
            categories = json_data.get("Categories", [])
            if categories:
                summary.append("ASSESSMENT STRUCTURE:")
                
                # Get parts information
                parts_info = self.get_category_parts_info(categories)
                
                for category in categories:
                    cat_name = category.get("Category", "")
                    weight = category.get("Weight", "")
                    part = category.get("Part", "1")
                    
                    if cat_name:
                        # Format the category heading
                        if part != "1":
                            # For multiple parts, show total contribution
                            try:
                                part_count = int(part)
                                weight_value = float(weight.replace("%", ""))
                                total = weight_value * part_count
                                cat_line = f"- {cat_name} ({part} parts × {weight} = {total:.1f}% total)"
                            except (ValueError, ZeroDivisionError):
                                cat_line = f"- {cat_name} ({weight}) - {part} parts"
                        else:
                            cat_line = f"- {cat_name} ({weight})"
                                
                        summary.append(cat_line)
                        
                        # Add details about the category
                        clo = category.get("CLO", "")
                        if clo:
                            summary.append(f"  Learning outcomes: {clo}")
                            
                        question_type = category.get("Question Type", "")
                        if question_type:
                            summary.append(f"  Format: {question_type}")
                            
                        note = category.get("Note", "")
                        if note:
                            # Clean up multi-line notes
                            clean_note = note.replace("\n", " ")
                            summary.append(f"  Note: {clean_note}")
                            
                        summary.append("")
            
            # Session structure - simplified
            sessions = json_data.get("StudentMaterials", [])
            if sessions:
                # Count quiz sessions
                quiz_sessions = [s for s in sessions if "Quiz" in s.get("Topic", "")]
                milestone_sessions = [s for s in sessions if "Milestone" in s.get("Topic", "")]
                
                summary.append("SESSION STRUCTURE:")
                summary.append(f"- Total sessions: {len(sessions)}")
                summary.append(f"- Quiz sessions: {len(quiz_sessions)}")
                summary.append(f"- Milestone presentation sessions: {len(milestone_sessions)}")
                summary.append("")
            
            return "\n".join(summary)
            
        except Exception as e:
            print(f"Error creating syllabus summary: {e}")
            return f"Error creating summary for {json_data.get('Subject Code', 'UNKNOWN')}"
            
    def process_syllabus_file(self, file_path, output_format="plain_text"):
        """
        Process a syllabus JSON file and convert to the requested format
        
        Args:
            file_path: Path to JSON file
            output_format: Format to convert to ("plain_text" or "summary")
            
        Returns:
            str: Processed syllabus in the requested format
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            if output_format == "summary":
                return self.create_syllabus_summary(json_data)
            else:
                return self.convert_syllabus_to_plain_text(json_data)

        except Exception as e:
            print(f"Error processing syllabus file {file_path}: {e}")
            return ""

    def convert_syllabus_to_intelligent_chunks(self, json_data):
        """
        Convert course syllabus JSON data to intelligent chunks optimized for LLM.
        Each chunk focuses on a specific aspect and includes the course code.

        Args:
            json_data: Parsed JSON data of a course syllabus

        Returns:
            list: List of Document objects with intelligent chunking
        """
        try:
            subject_code = json_data.get("Subject Code", "")
            if not subject_code:
                return self._create_fallback_chunks(json_data)

            chunks = []

            # Chunk 1: Basic Course Information
            basic_info_chunk = self._create_basic_info_chunk(json_data, subject_code)
            if basic_info_chunk:
                chunks.append(basic_info_chunk)

            # Chunk 2: Learning Outcomes (CLOs)
            clo_chunk = self._create_clo_chunk(json_data, subject_code)
            if clo_chunk:
                chunks.append(clo_chunk)

            # Chunk 3: Course Materials
            materials_chunk = self._create_materials_chunk(json_data, subject_code)
            if materials_chunk:
                chunks.append(materials_chunk)

            # Chunk 4: Assessment Structure
            assessment_chunk = self._create_assessment_chunk(json_data, subject_code)
            if assessment_chunk:
                chunks.append(assessment_chunk)

            # Chunks 5+: Session Groups (every 5 sessions)
            session_chunks = self._create_session_chunks(json_data, subject_code, sessions_per_chunk=5)
            chunks.extend(session_chunks)

            return chunks

        except Exception as e:
            print(f"Error creating intelligent chunks: {e}")
            return self._create_fallback_chunks(json_data)

    def _create_basic_info_chunk(self, json_data, subject_code):
        """Create a chunk with basic course information"""
        try:
            content_parts = [f"Course Code: {subject_code}"]

            # Course name
            syllabus_name = json_data.get("Syllabus Name", "")
            if syllabus_name:
                content_parts.append(f"Course Name: {syllabus_name}")

            # Credits
            credits = json_data.get("NoCredit", "")
            if credits:
                content_parts.append(f"Credits: {credits}")

            # Degree level
            degree_level = json_data.get("Degree Level", "")
            if degree_level:
                content_parts.append(f"Degree Level: {degree_level}")

            # Time allocation
            time_allocation = json_data.get("Time Allocation", "")
            if time_allocation:
                content_parts.append(f"Time Allocation: {time_allocation}")

            # Prerequisites
            prerequisite = json_data.get("Pre-Requisite", "")
            if prerequisite:
                content_parts.append(f"Prerequisites: {prerequisite}")

            # Course description
            description = json_data.get("Description", "")
            if description:
                content_parts.append(f"Course Description: {description}")

            # Student tasks
            student_tasks = json_data.get("StudentTasks", "")
            if student_tasks:
                content_parts.append(f"Student Requirements: {student_tasks}")

            # Tools required
            tools = json_data.get("Tools", "")
            if tools:
                content_parts.append(f"Required Tools: {tools}")

            # Minimum Average Mark To Pass
            min_avg_mark = json_data.get("MinAvgMarkToPass", "")
            if min_avg_mark:
                content_parts.append(f"Minimum Average Mark To Pass: {min_avg_mark}")

            # Scoring Scale
            scoring_scale = json_data.get("Scoring Scale", "")
            if scoring_scale:
                content_parts.append(f"Scoring Scale: {scoring_scale}")

            content = "\n\n".join(content_parts)

            return Document(
                page_content=content,
                metadata={
                    "course_code": subject_code,
                    "chunk_type": "basic_info",
                    "title": f"{subject_code} - Basic Course Information"
                }
            )

        except Exception as e:
            print(f"Error creating basic info chunk: {e}")
            return None

    def _create_clo_chunk(self, json_data, subject_code):
        """Create a chunk with Course Learning Outcomes (CLOs)"""
        try:
            clos = json_data.get("CLOs", [])
            if not clos:
                return None

            content_parts = [f"Course Code: {subject_code}", "Course Learning Outcomes (CLOs):"]

            for clo in clos:
                clo_name = clo.get("CLO Name", "")
                lo_details = clo.get("LO Details", "")

                if clo_name and lo_details:
                    content_parts.append(f"CLO{clo_name}: {lo_details}")

            content = "\n\n".join(content_parts)

            return Document(
                page_content=content,
                metadata={
                    "course_code": subject_code,
                    "chunk_type": "learning_outcomes",
                    "title": f"{subject_code} - Learning Outcomes",
                    "clo_count": len(clos)
                }
            )

        except Exception as e:
            print(f"Error creating CLO chunk: {e}")
            return None

    def _create_materials_chunk(self, json_data, subject_code):
        """Create a chunk with course materials"""
        try:
            materials = json_data.get("Materials", [])
            if not materials:
                return None

            content_parts = [f"Course Code: {subject_code}", "Course Materials:"]

            main_materials = []
            supplementary_materials = []

            for material in materials:
                description = material.get("MaterialDescription", "")
                author = material.get("Author", "")
                publisher = material.get("Publisher", "")
                published_date = material.get("PublishedDate", "")
                edition = material.get("Edition", "")
                isbn = material.get("ISBN", "")
                is_main = material.get("IsMainMaterial", False)
                is_online = material.get("IsOnline", False)
                note = material.get("Note", "")

                if description:
                    material_info = [f"Title: {description}"]

                    if author:
                        material_info.append(f"Author: {author}")
                    if publisher:
                        material_info.append(f"Publisher: {publisher}")
                    if published_date:
                        material_info.append(f"Year: {published_date}")
                    if edition:
                        material_info.append(f"Edition: {edition}")
                    if isbn:
                        material_info.append(f"ISBN: {isbn}")

                    format_type = "Online" if is_online else "Physical"
                    material_info.append(f"Format: {format_type}")

                    if note:
                        material_info.append(f"Note: {note}")

                    material_text = " | ".join(material_info)

                    if is_main:
                        main_materials.append(material_text)
                    else:
                        supplementary_materials.append(material_text)

            if main_materials:
                content_parts.append("Main Materials:")
                content_parts.extend([f"- {mat}" for mat in main_materials])

            if supplementary_materials:
                content_parts.append("Supplementary Materials:")
                content_parts.extend([f"- {mat}" for mat in supplementary_materials])

            content = "\n\n".join(content_parts)

            return Document(
                page_content=content,
                metadata={
                    "course_code": subject_code,
                    "chunk_type": "materials",
                    "title": f"{subject_code} - Course Materials",
                    "main_materials_count": len(main_materials),
                    "supplementary_materials_count": len(supplementary_materials)
                }
            )

        except Exception as e:
            print(f"Error creating materials chunk: {e}")
            return None

    def _create_assessment_chunk(self, json_data, subject_code):
        """Create a chunk with assessment structure with improved grade terminology"""
        try:
            categories = json_data.get("Categories", [])
            if not categories:
                return None

            content_parts = [f"Course Code: {subject_code}", "Assessment Grade Structure:"]

            total_weight = 0
            assessment_details = []
            has_passing_requirements = False

            for category in categories:
                cat_name = category.get("Category", "")
                weight = category.get("Weight", "")
                part = category.get("Part", "1")
                type_ = category.get("Type ", "")
                completion_criteria = category.get("Completion Criteria", "")
                clo = category.get("CLO", "")
                question_type = category.get("Question Type", "")
                no_question = category.get("No Question", "")
                note = category.get("Note", "")

                if cat_name and weight:
                    detail_parts = [f"Grade Assessment: {cat_name}"]

                    # Display weight as is, without multiplying by parts
                    detail_parts.append(f"Weight: {weight}")

                    # Add parts information if multiple parts exist
                    try:
                        parts_count = int(part) if part and part != "1" else 1
                        if parts_count > 1:
                            detail_parts.append(f"Parts: {parts_count}")

                        # Add to total weight (just the original weight value)
                        weight_value = float(weight.replace("%", ""))
                        total_weight += weight_value
                    except (ValueError, TypeError):
                        if part != "1":
                            detail_parts.append(f"Parts: {part}")

                    if type_:
                        detail_parts.append(f"Type: {type_}")

                    if completion_criteria:
                        has_passing_requirements = True
                        # Replace > with "Over" for better readability
                        if ">" in completion_criteria:
                            criteria_text = completion_criteria.replace(">", "Over ")
                            detail_parts.append(f"Minimum Score to Pass: {criteria_text}")
                        else:
                            detail_parts.append(f"Minimum Score to Pass: {completion_criteria}")

                    if clo:
                        detail_parts.append(f"Evaluates CLOs: {clo}")

                    if question_type:
                        detail_parts.append(f"Format: {question_type}")

                    if no_question:
                        detail_parts.append(f"Number of Questions: {no_question}")

                    if note:
                        clean_note = note.replace("\n", " ").strip()
                        detail_parts.append(f"Notes: {clean_note}")

                    assessment_details.append(" | ".join(detail_parts))

            # Add assessment details
            for detail in assessment_details:
                content_parts.append(f"- {detail}")

            if total_weight > 0:
                content_parts.append(f"\nTotal Grade Weight: {total_weight}%")

            # Add passing requirement clarification
            if has_passing_requirements:
                content_parts.append(f"\nIMPORTANT PASSING REQUIREMENT for {subject_code}: Students must pass ALL assessment components (meet minimum score requirements for each component) to be eligible to pass the course. Simply achieving the overall course average is not sufficient - each individual assessment must meet its minimum passing criteria.")

            content = "\n\n".join(content_parts)

            return Document(
                page_content=content,
                metadata={
                    "course_code": subject_code,
                    "chunk_type": "assessment",
                    "title": f"{subject_code} - Grade Assessment Structure",
                    "assessment_count": len(assessment_details),
                    "total_weight": total_weight
                }
            )

        except Exception as e:
            print(f"Error creating assessment chunk: {e}")
            return None

    def _create_session_chunks(self, json_data, subject_code, sessions_per_chunk=10):
        """Create chunks for course sessions, grouped for better context"""
        try:
            sessions = json_data.get("StudentMaterials", [])
            if not sessions:
                return []

            chunks = []

            # Group sessions into chunks
            for i in range(0, len(sessions), sessions_per_chunk):
                session_group = sessions[i:i + sessions_per_chunk]

                content_parts = [f"Course Code: {subject_code}"]

                start_session = i + 1
                end_session = min(i + sessions_per_chunk, len(sessions))
                content_parts.append(f"Sessions {start_session}-{end_session}:")

                for session in session_group:
                    session_num = session.get("Session", "")
                    topic = session.get("Topic", "")
                    learning_type = session.get("Learning-Teaching Type", "")
                    lo = session.get("LO", "")
                    student_tasks = session.get("Student's Tasks", "")
                    materials = session.get("Student Materials", "")

                    if session_num and topic:
                        session_parts = [f"Session {session_num}: {topic}"]

                        if learning_type:
                            session_parts.append(f"Type: {learning_type}")

                        if lo:
                            session_parts.append(f"Learning Outcomes: {lo}")

                        if student_tasks:
                            session_parts.append(f"Tasks: {student_tasks}")

                        if materials:
                            session_parts.append(f"Materials: {materials}")

                        content_parts.append(" | ".join(session_parts))

                content = "\n\n".join(content_parts)

                chunk = Document(
                    page_content=content,
                    metadata={
                        "course_code": subject_code,
                        "chunk_type": "sessions",
                        "title": f"{subject_code} - Sessions {start_session}-{end_session}",
                        "session_start": start_session,
                        "session_end": end_session,
                        "session_count": len(session_group)
                    }
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Error creating session chunks: {e}")
            return []

    def _create_fallback_chunks(self, json_data):
        """Create simple fallback chunks if intelligent chunking fails"""
        try:
            subject_code = json_data.get("Subject Code", "UNKNOWN")
            content = json.dumps(json_data, indent=2, ensure_ascii=False)

            return [Document(
                page_content=f"Course Code: {subject_code}\n\n{content}",
                metadata={
                    "course_code": subject_code,
                    "chunk_type": "fallback",
                    "title": f"{subject_code} - Complete Syllabus Data"
                }
            )]

        except Exception as e:
            print(f"Error creating fallback chunks: {e}")
            return []

    def process_syllabus_to_chunks(self, file_path):
        """
        Process a syllabus JSON file and convert to intelligent chunks

        Args:
            file_path: Path to JSON file

        Returns:
            list: List of Document objects with intelligent chunking
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            return self.convert_syllabus_to_intelligent_chunks(json_data)

        except Exception as e:
            print(f"Error processing syllabus file {file_path}: {e}")
            return []

# For testing
if __name__ == "__main__":
    converter = SyllabusConverter()
    test_file = "ADS301m.json"
    if os.path.exists(test_file):
        # Test syllabus detection
        print(f"Is {test_file} a syllabus file? {converter.is_syllabus_file(test_file)}")

        # Test plain text conversion
        plain_text = converter.process_syllabus_to_chunks(test_file)
        print("\nPLAIN TEXT CONVERSION:")
        print(f"Total length: {len(plain_text)} characters")
        print(plain_text)  # Show first 500 chars

        # Test summary format
        summary = converter.process_syllabus_file(test_file, output_format="summary")
        print("\nSUMMARY FORMAT:")
        print(summary)
