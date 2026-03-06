package com.example.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    Optional<User> findByEmail(String email);

    List<User> findByActiveTrue();

    @Query("SELECT u FROM User u WHERE u.name LIKE %:name%")
    List<User> searchByName(String name);

    boolean existsByEmail(String email);
}
